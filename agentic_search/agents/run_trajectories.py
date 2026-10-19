"""Run source-only adaptive trajectories through the isolated MCP tools."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from agentic_search.agents.model_cli import call_model, call_static, _mcp_tool_call, _tool_env
from agentic_search.evaluator.source_evaluator import C_GRID


PACKAGE = Path(__file__).resolve().parents[1]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select_robust(history: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(history, key=lambda row: (-float(row["min_source_auc"]), str(row["quartet_key"])))[0]


def _realize_user_ids(quartet: list[str], opaque: bool) -> list[str]:
    if not opaque:
        return sorted(quartet)
    mapping = pd.read_csv(PACKAGE / "data/opaque_head_mapping.csv", dtype=str)
    lookup = dict(zip(mapping.opaque_id, mapping.head_id))
    return sorted(lookup[value] for value in quartet)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _model_catalog(opaque: bool) -> list[dict[str, str]]:
    catalog = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str)
    if not opaque:
        return [{"head_id": str(row.head_id), "head_name": str(row.head_name),
                 "taxonomy_category": str(row.taxonomy_category)}
                for row in catalog.itertuples(index=False)]
    mapping = pd.read_csv(PACKAGE / "data/opaque_head_mapping.csv", dtype=str)
    return [{"head_id": str(value)} for value in sorted(mapping.opaque_id.astype(str))]


def _model_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = ("auc_source1_to_source2", "auc_source2_to_source1", "mean_source_auc",
               "min_source_auc", "direction_gap", "selected_C")
    return [{"head_ids": list(map(str, row["head_ids"])),
             **{key: float(row[key]) for key in allowed}}
            for row in history]


def _proposal_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {
        "decision_summary": {"type": "string", "maxLength": 1000},
        "candidates": {"type": "array", "minItems": 5, "maxItems": 5,
                        "items": {"type": "array", "minItems": 4, "maxItems": 4,
                                  "items": {"type": "string"}}}},
        "required": ["decision_summary", "candidates"], "additionalProperties": False}


def _choice_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {
        "final_choice": {"type": "array", "minItems": 4, "maxItems": 4,
                          "items": {"type": "string"}},
        "decision_summary": {"type": "string", "maxLength": 1000}},
        "required": ["final_choice", "decision_summary"], "additionalProperties": False}


def _codex_controller_round(*, model: str, system_prompt: str, user_prompt: str,
                            state_path: Path, held_site: str, round_index: int,
                            seed: int, budget: int, opaque: bool,
                            final_round: bool,
                            reasoning_effort: str | None) -> dict[str, Any]:
    """Keep Codex tool-free; the controller supplies only frozen catalog/source results."""
    environment = _tool_env(state_path.resolve(), held_site, round_index, seed, budget, opaque)
    catalog = _model_catalog(opaque)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = _model_history(state.get("history", []))
    context = (user_prompt + "\n\nFrozen catalog (JSON; IDs and permitted descriptions only):\n" +
               json.dumps(catalog, ensure_ascii=False, separators=(",", ":")) +
               "\n\nComplete evaluated history in chronological order (source results only):\n" +
               json.dumps(history, separators=(",", ":"), allow_nan=False) +
               "\n\nReturn a new batch of exactly five candidates now. Do not repeat any quartet in the history.")
    proposal_calls: list[dict[str, Any]] = []
    proposal_answer: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    for attempt in range(2):
        prompt = context
        if attempt:
            prompt += ("\n\nYour previous batch was rejected. Repair it using only this controller feedback: " +
                       json.dumps(evaluation.get("errors", []), ensure_ascii=False) +
                       ". Return exactly five valid, distinct, never-before-evaluated quartets from the catalog.")
        proposal = call_static("codex", model, system_prompt, prompt, _proposal_schema(),
                               seed=int(seed) + round_index * 1009 + attempt,
                               reasoning_effort=reasoning_effort)
        proposal_answer = proposal["answer"]
        evaluation = _mcp_tool_call("evaluate_batch", {
            "candidates": proposal_answer.get("candidates", [])}, environment)
        proposal_calls.append({"attempt": attempt + 1, "metadata": proposal["metadata"],
                               "decision_summary": proposal_answer.get("decision_summary", ""),
                               "candidates": proposal_answer.get("candidates", []),
                               "controller_result": evaluation})
        if evaluation.get("valid") is not True:
            _append_jsonl(state_path.parent / "invalid_proposal_attempts.jsonl", {
                "trajectory_id": state.get("experiment_id"), "provider": "codex",
                "model": model, "held_site": held_site, "budget": budget, "seed": seed,
                "round": round_index, "attempt": attempt + 1,
                "candidates": proposal_answer.get("candidates", []),
                "controller_result": evaluation,
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            })
        if evaluation.get("valid") is True:
            break
    if not evaluation or evaluation.get("valid") is not True or proposal_answer is None:
        raise RuntimeError("Controller did not complete the source-only batch after one repair")

    answer = {"decision_summary": str(proposal_answer.get("decision_summary", "")),
              "candidates": proposal_answer.get("candidates", [])}
    choice_call = None
    choice_summary = None
    if final_round:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        final_history = _model_history(state.get("history", []))
        choice_prompt = (
            "The final five candidates have now been evaluated, completing your source-only budget. "
            "Select exactly one quartet from the full evaluated history below. Prefer a candidate with a "
            "strong minimum across the two source directions and a small directional gap; consider the "
            "feature descriptions and your scientific judgment. You still have no target-site results.\n\n"
            "Complete evaluated history (source results only):\n" +
            json.dumps(final_history, separators=(",", ":"), allow_nan=False) +
            "\n\nReturn the selected four IDs and a brief rationale."
        )
        choice_call = call_static("codex", model, system_prompt, choice_prompt, _choice_schema(),
                                  seed=int(seed) + round_index * 1009 + 997,
                                  reasoning_effort=reasoning_effort)
        answer["final_choice"] = choice_call["answer"].get("final_choice", [])
        choice_summary = str(choice_call["answer"].get("decision_summary", ""))
        if choice_summary:
            answer["decision_summary"] = choice_summary

    metadata = {"provider": "codex", "model_requested": model, "model_reported": model,
                "cli_version": proposal_calls[-1]["metadata"].get("cli_version"),
                "mcp_tool_approval": "controller_only_source_evaluator",
                "source_evaluation_controller": "local evaluate_batch; target code is not exposed",
                "reasoning_effort": reasoning_effort,
                "temperature_controlled": False,
                "controller_proposal_calls": [row["metadata"] for row in proposal_calls],
                "controller_attempts": proposal_calls,
                "final_choice_call": choice_call["metadata"] if choice_call else None,
                "controller_repair_attempts": len(proposal_calls) - 1}
    return {"answer": answer, "metadata": metadata, "choice_summary": choice_summary,
            "cli_stderr_tail": ""}


def run_trajectory(*, provider: str, model: str, held_site: str, budget: int, seed: int,
                   method: str = "full_agent", prompt_name: str | None = None,
                   output_dir: Path | None = None) -> dict[str, Any]:
    opaque = method == "opaque_iterative"
    prompt_file = PACKAGE / "configs/prompts" / (prompt_name or ("opaque_agent.txt" if opaque else "full_agent.txt"))
    system_prompt = prompt_file.read_text(encoding="utf-8")
    protocol_freeze = json.loads((PACKAGE / "runs/protocol_freeze.json").read_text(encoding="utf-8"))
    protocol_hash = protocol_freeze["protocol_sha256"]
    prompt_hash = _hash_file(prompt_file)
    protocol = yaml.safe_load((PACKAGE / "configs/protocol.yaml").read_text(encoding="utf-8"))
    configured = next((row for row in protocol.get("agent_models", [])
                      if row.get("provider") == provider and row.get("model") == model), None)
    if configured is None and method not in {"pilot", "mcp_integration_check"}:
        raise ValueError("Provider/model pair is not in the frozen protocol")
    reasoning_effort = (configured.get("reasoning_effort") if configured else None) or protocol.get("reasoning_effort")
    if int(budget) not in {int(x) for x in protocol["main_budgets"]}:
        raise ValueError("Full adaptive trajectory budget is not in the frozen main grid")
    if budget % int(protocol["batch_size"]):
        raise ValueError("Every adaptive budget must be a multiple of the batch size")
    trajectory_id = f"{method}__{provider}-{model}__{held_site}__B{budget}__seed{seed}"
    root = output_dir or PACKAGE / "runs/source_only/agents"
    state_path = root / method / provider / held_site / f"B{budget}_seed{seed}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    resumed = False
    if state_path.exists():
        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
        if (state.get("budget_used") == budget and state.get("completed_at_utc")
                and state.get("robust_selected_quartet_key")):
            return state
        if state.get("experiment_id") != trajectory_id or int(state.get("budget", -1)) != int(budget):
            raise ValueError(f"Existing trajectory state does not match this run: {state_path}")
        if state.get("protocol_sha256") != protocol_hash or state.get("prompt_sha256") != prompt_hash:
            raise ValueError(f"Cannot resume under a changed protocol or prompt: {state_path}")
        if int(state.get("budget_used", -1)) != len(state.get("history", [])) or int(state["budget_used"]) % 5:
            raise ValueError(f"Partial trajectory has inconsistent budget/history state: {state_path}")
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        state.pop("failed_provider_responses", None)
        state.pop("failed_at_round", None)
        resumed = True
    else:
        state = {
            "experiment_id": trajectory_id, "method": method,
            "provider": provider, "model_requested": model,
            "held_site_internal": held_site, "budget": int(budget), "seed": int(seed),
            "budget_used": 0, "round": 1, "repair_attempts": 0,
            "history": [], "invalid_attempts": [], "auto_repair": False,
            "protocol_sha256": protocol_hash, "prompt_sha256": prompt_hash,
            "temperature_requested": float(protocol["agent_temperature"]),
            "temperature_controlled": provider == "ollama",
            "temperature_applied": 0.2 if provider == "ollama" else None,
            "reasoning_effort": reasoning_effort,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    round_records = []
    rounds = int(budget) // int(protocol["batch_size"])
    start_round = int(state["budget_used"]) // int(protocol["batch_size"]) + 1
    for round_index in range(start_round, rounds + 1):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["round"] = round_index
        state["round_evaluated"] = False
        state["repair_attempts"] = 0
        state.pop("last_batch", None)
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        user_prompt = (
            f"This is round {round_index} of {rounds}. The total source-evaluation budget is {budget}; "
            f"{state['budget_used']} have been used and {budget - state['budget_used']} remain. "
            "First call get_history so you can use all prior evidence. Use search_heads if useful. "
            "Call evaluate_batch exactly once with five new quartets. If it reports a validation error, "
            "repair the batch once. After a successful result, return the required JSON. "
            + ("This is the final round: also choose one quartet from the complete history as final_choice. "
               if round_index == rounds else "")
            + "Do not request or use target-site information."
        )
        prior_invalid = [row for row in state.get("invalid_attempts", []) if int(row.get("round", -1)) == round_index]
        if prior_invalid:
            user_prompt += " Previous batch validation feedback to address: " + json.dumps(prior_invalid[-1])
        attempt_outputs = []
        valid = False
        for attempt in range(2):
            if attempt:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                feedback = state.get("invalid_attempts", [])[-1:] or [{"errors": ["No successful evaluate_batch call was recorded."]}]
                user_prompt += " Your previous attempt was not accepted; validation feedback: " + json.dumps(feedback)
            try:
                if provider == "codex":
                    call = _codex_controller_round(
                        model=model, system_prompt=system_prompt, user_prompt=user_prompt,
                        state_path=state_path, held_site=held_site, round_index=round_index,
                        seed=seed, budget=budget, opaque=opaque, final_round=(round_index == rounds),
                        reasoning_effort=reasoning_effort)
                else:
                    call = call_model(provider, model, system_prompt, user_prompt,
                                      state_path=state_path, held_site=held_site, round_index=round_index,
                                      seed=seed, budget=budget, opaque=opaque, final_round=(round_index == rounds))
            except Exception as exc:
                attempt_outputs.append({"provider_error": f"{type(exc).__name__}: {str(exc)[:1200]}"})
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("round_evaluated") and state.get("last_batch"):
                    synthesized = {"decision_summary": "Source batch completed; final response recovered after provider interruption.",
                                   "candidates": [row["head_ids"] for row in state["last_batch"]]}
                    if round_index == rounds:
                        synthesized["final_choice"] = []
                    call = {"answer": synthesized,
                            "metadata": {"provider": provider, "model_requested": model,
                                         "recovered_after_provider_error": True,
                                         "temperature_controlled": provider == "ollama",
                                         "temperature": 0.2 if provider == "ollama" else None}}
                    state["provider_response_recovered"] = True
                elif attempt == 0:
                    user_prompt += " Retry the same round. Use the frozen source tools and submit a valid five-quartet batch."
                    continue
                else:
                    break
            state = json.loads(state_path.read_text(encoding="utf-8"))
            answer = call["answer"]
            eval_done = bool(state.get("round_evaluated"))
            echoed = [sorted(map(str, q)) for q in answer.get("candidates", [])]
            submitted = [sorted(map(str, row["head_ids"])) for row in state.get("last_batch", [])] if eval_done else []
            echo_matches = eval_done and echoed == submitted
            attempt_outputs.append({"metadata": call["metadata"], "decision_summary": answer.get("decision_summary", ""),
                                    "candidates": answer.get("candidates", []),
                                    "cli_stderr_tail": call.get("cli_stderr_tail", ""),
                                    "echo_matches_tool_batch": bool(echo_matches),
                                    "validation_errors_before_call": state.get("invalid_attempts", [])[-1:]
                                    if state.get("invalid_attempts", []) and int(state["invalid_attempts"][-1]["round"]) == round_index else []})
            if eval_done:
                valid = True
                if not echo_matches:
                    state["model_echo_matches_evaluate_batch"] = False
                else:
                    state["model_echo_matches_evaluate_batch"] = True
                state["last_decision_summary"] = str(answer.get("decision_summary", ""))
                state["last_provider_metadata"] = call["metadata"]
                if call.get("choice_summary"):
                    state["agent_choice_summary"] = call["choice_summary"]
                if round_index == rounds:
                    candidate_choice = answer.get("final_choice")
                    try:
                        if not isinstance(candidate_choice, list) or len(candidate_choice) != 4 or len(set(candidate_choice)) != 4:
                            raise ValueError("final_choice must contain four unique IDs")
                        choice_key = "|".join(sorted(map(str, candidate_choice)))
                        choices = {row["quartet_key"] for row in state["history"]}
                        if choice_key not in choices:
                            raise ValueError("final_choice is not among this trajectory's evaluated candidates")
                        state["agent_choice_quartet_key"] = choice_key
                        state["agent_choice_quartet_real_head_ids"] = _realize_user_ids(list(map(str, candidate_choice)), opaque)
                        state["agent_choice_valid"] = True
                    except Exception as exc:
                        state["agent_choice_valid"] = False
                        state["agent_choice_error"] = str(exc)
                state["rounds_completed"] = round_index
                state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                break
        if not valid:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["failed_provider_responses"] = attempt_outputs
            state["failed_at_round"] = round_index
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise RuntimeError(f"Model failed to submit a valid batch for {trajectory_id}, round {round_index}; see {state_path}")
        round_row = {
            "experiment_id": trajectory_id, "method": method, "provider": provider,
            "model_requested": model, "held_site_internal": held_site,
            "budget": int(budget), "trajectory_seed": int(seed), "round": round_index,
            "budget_used_before": int(state["budget_used"] - 5),
            "decision_summary": state.get("last_decision_summary", ""),
            "proposed_candidates": [row["head_ids"] for row in state["last_batch"]],
            "source_results": state["last_batch"],
            "invalid_attempts": len(state.get("invalid_attempts", [])),
            "auto_repair": any(bool(row.get("auto_repair")) for row in state["last_batch"]),
            "prompt_hash": prompt_hash, "protocol_hash": protocol_hash,
            "provider_calls": attempt_outputs,
        }
        _append_jsonl(root / "logs" / f"{method}_{provider}_{held_site}_B{budget}_seed{seed}.jsonl", round_row)
        print(f"{trajectory_id}: round {round_index}/{rounds}, source budget {state['budget_used']}/{budget}", flush=True)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("budget_used", -1)) != int(budget) or len(state.get("history", [])) != int(budget):
        raise AssertionError("Agent trajectory did not consume its exact source-evaluation budget")
    robust = _select_robust(state["history"])
    state["robust_selected_quartet_key"] = robust["quartet_key"]
    state["robust_selected_real_head_ids"] = _realize_user_ids(robust["head_ids"], opaque)
    state["robust_selected_source_metrics"] = {key: robust[key] for key in
        ("mean_source_auc", "min_source_auc", "direction_gap", "selected_C")}
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = root / "trajectories.parquet"
    summary = {
        "experiment_id": trajectory_id, "method": method, "provider": provider,
        "model_requested": model, "model_reported": state.get("last_provider_metadata", {}).get("model_reported"),
        "held_site": held_site, "budget": int(budget), "seed": int(seed),
        "n_evaluations": len(state["history"]), "robust_selected_quartet_key": state["robust_selected_quartet_key"],
        "robust_selected_real_head_ids": state["robust_selected_real_head_ids"],
        "source_mean": robust["mean_source_auc"], "source_min": robust["min_source_auc"],
        "source_gap": robust["direction_gap"], "selected_C": robust["selected_C"],
        "agent_choice_quartet_key": state.get("agent_choice_quartet_key"),
        "agent_choice_real_head_ids": state.get("agent_choice_quartet_real_head_ids"),
        "agent_choice_valid": state.get("agent_choice_valid", False),
        "auto_repair": state.get("auto_repair", False),
        "invalid_attempt_count": len(state.get("invalid_attempts", [])),
        "protocol_sha256": protocol_hash, "prompt_sha256": prompt_hash,
        "temperature_requested": state["temperature_requested"],
        "temperature_controlled": bool(state.get("last_provider_metadata", {}).get("temperature_controlled", False)),
        "temperature_applied": state.get("last_provider_metadata", {}).get("temperature"),
        "reasoning_effort": state.get("last_provider_metadata", {}).get("reasoning_effort"),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.with_suffix(".lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing_rows = pd.read_parquet(summary_path).to_dict("records") if summary_path.exists() else []
        existing_rows = [row for row in existing_rows if row.get("experiment_id") != trajectory_id]
        existing_rows.append(summary)
        temporary = summary_path.with_suffix(".tmp.parquet")
        pd.DataFrame(existing_rows).sort_values("experiment_id").to_parquet(temporary, index=False)
        temporary.replace(summary_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return state


def run_pilot(provider: str, model: str) -> None:
    for seed in range(3000, 3005):
        run_trajectory(provider=provider, model=model, held_site="I0002", budget=25,
                       seed=seed, method="pilot", prompt_name="full_agent.txt",
                       output_dir=PACKAGE / "runs/pilot")


def run_grid(provider: str, model: str, *, sites: list[str] | None = None,
             budgets: list[int] | None = None, seeds: list[int] | None = None,
             method: str = "full_agent", workers: int = 1) -> None:
    protocol = yaml.safe_load((PACKAGE / "configs/protocol.yaml").read_text(encoding="utf-8"))
    folds = json.loads((PACKAGE / "data/folds.json").read_text(encoding="utf-8"))
    sites = sites or folds["site_order"]
    budgets = budgets or protocol["main_budgets"]
    seeds = seeds or protocol["trajectory_seeds"]
    jobs = [(held_site, int(budget), int(seed)) for held_site in sites
            for budget in budgets for seed in seeds]
    if int(workers) <= 1:
        for held_site, budget, seed in jobs:
            run_trajectory(provider=provider, model=model, held_site=held_site,
                           budget=budget, seed=seed, method=method)
        return
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(run_trajectory, provider=provider, model=model,
                               held_site=held_site, budget=budget, seed=seed, method=method):
                   (held_site, budget, seed)
                   for held_site, budget, seed in jobs}
        for future in as_completed(futures):
            held_site, budget, seed = futures[future]
            future.result()
            print(f"Completed {method} {held_site} B{budget} seed{seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("codex", "claude", "ollama"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--method", choices=("full_agent", "opaque_iterative"), default="full_agent")
    parser.add_argument("--site", action="append")
    parser.add_argument("--budget", action="append", type=int)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent trajectories to run concurrently; each trajectory remains isolated")
    args = parser.parse_args()
    if args.pilot:
        run_pilot(args.provider, args.model)
    else:
        run_grid(args.provider, args.model, sites=args.site, budgets=args.budget,
                 seeds=args.seed, method=args.method, workers=args.workers)


if __name__ == "__main__":
    main()
