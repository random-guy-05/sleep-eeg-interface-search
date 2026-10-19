"""Run preregistered source-only GPT-6 Luna / Opus 5.5 controller experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import agentic_search.agents.model_cli as model_cli
import agentic_search.agents.run_trajectories as trajectories
from agentic_search.agents.run_trajectories import run_trajectory
from agentic_search.evaluator.source_evaluator import evaluate_source

PACKAGE = Path(__file__).resolve().parents[2]
SITES = ("I0002", "I0006", "S0001")
SEEDS = tuple(range(1000, 1010))
PROMPT = PACKAGE / "configs/prompts/full_agent.txt"
CATALOG = PACKAGE / "data/heads.csv"
OUT = PACKAGE / "runs/supplemental/generalizability"
LEDGER = OUT / "provider_usage.jsonl"
_ledger_lock = threading.Lock()
_original_run = subprocess.run


def _claude_controller_round(provider, model, system_prompt, user_prompt, *, state_path, held_site,
                             round_index, seed, budget, opaque, final_round):
    """Claude counterpart to Sol's static/controller-mediated source evaluator."""
    from agentic_search.agents.model_cli import call_static, _mcp_tool_call, _tool_env
    from agentic_search.agents.run_trajectories import (
        _choice_schema, _model_catalog, _model_history, _proposal_schema,
    )

    environment = _tool_env(state_path.resolve(), held_site, round_index, seed, budget, opaque)
    catalog = _model_catalog(opaque)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = _model_history(state.get("history", []))
    context = (user_prompt + "\n\nFrozen catalog (JSON; IDs and permitted descriptions only):\n" +
               json.dumps(catalog, ensure_ascii=False, separators=(",", ":")) +
               "\n\nComplete evaluated history in chronological order (source results only):\n" +
               json.dumps(history, separators=(",", ":"), allow_nan=False) +
               "\n\nReturn a new batch of exactly five candidates now. Do not repeat any quartet in the history.")
    proposal_calls = []
    evaluation = None
    proposal_answer = None
    for attempt in range(2):
        prompt = context
        if attempt:
            prompt += ("\n\nYour previous batch was rejected. Repair it using only this controller feedback: " +
                       json.dumps(evaluation.get("errors", []), ensure_ascii=False) +
                       ". Return exactly five valid, distinct, never-before-evaluated quartets from the catalog.")
        proposal = call_static("claude", model, system_prompt, prompt, _proposal_schema(),
                               seed=int(seed) + round_index * 1009 + attempt)
        proposal_answer = proposal["answer"]
        evaluation = _mcp_tool_call("evaluate_batch", {"candidates": proposal_answer.get("candidates", [])}, environment)
        proposal_calls.append({"attempt": attempt + 1, "metadata": proposal["metadata"],
                               "decision_summary": proposal_answer.get("decision_summary", ""),
                               "candidates": proposal_answer.get("candidates", []),
                               "controller_result": evaluation})
        if evaluation.get("valid") is not True:
            invalid_path = state_path.parent / "invalid_proposal_attempts.jsonl"
            invalid_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"trajectory_id": state.get("experiment_id"), "provider": "claude",
                   "model": model, "held_site": held_site, "budget": budget, "seed": seed,
                   "round": round_index, "attempt": attempt + 1,
                   "candidates": proposal_answer.get("candidates", []),
                   "controller_result": evaluation,
                   "captured_at_utc": datetime.now(timezone.utc).isoformat()}
            with _ledger_lock, invalid_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        if evaluation.get("valid") is True:
            break
    if not evaluation or evaluation.get("valid") is not True or proposal_answer is None:
        raise RuntimeError("Opus controller failed to submit a valid source batch after Sol's one-repair rule")

    answer = {"decision_summary": str(proposal_answer.get("decision_summary", "")),
              "candidates": proposal_answer.get("candidates", [])}
    choice_call, choice_summary = None, None
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
            "\n\nReturn the selected four IDs and a brief rationale.")
        choice_call = call_static("claude", model, system_prompt, choice_prompt, _choice_schema(),
                                  seed=int(seed) + round_index * 1009 + 997)
        answer["final_choice"] = choice_call["answer"].get("final_choice", [])
        choice_summary = str(choice_call["answer"].get("decision_summary", ""))
        if choice_summary:
            answer["decision_summary"] = choice_summary
    metadata = {"provider": "claude", "model_requested": model,
                "model_reported": proposal_calls[-1]["metadata"].get("model_reported"),
                "cli_version": proposal_calls[-1]["metadata"].get("cli_version"),
                "mcp_tool_approval": "controller_only_local_source_evaluator",
                "source_evaluation_controller": "local Sol-compatible evaluate_batch; target code not exposed",
                "reasoning_effort": "medium", "temperature_controlled": False,
                "controller_proposal_calls": [row["metadata"] for row in proposal_calls],
                "controller_attempts": proposal_calls,
                "final_choice_call": choice_call["metadata"] if choice_call else None,
                "controller_repair_attempts": len(proposal_calls) - 1}
    return {"answer": answer, "metadata": metadata, "choice_summary": choice_summary,
            "cli_stderr_tail": ""}


def _patch_claude_effort() -> None:
    """Force and audit explicit medium effort without editing frozen primary code."""
    def audited_run(args, *positional, **kwargs):
        command = list(args) if isinstance(args, (list, tuple)) else args
        provider_call = isinstance(command, list) and command and str(command[0]).endswith("claude")
        if provider_call and "--print" in command and "--effort" not in command:
            command = command[:1] + ["--effort", "medium"] + command[1:]
        result = _original_run(command, *positional, **kwargs)
        if provider_call and "--print" in command:
            metadata: dict = {"provider": "claude", "requested_model": "claude-opus-5-5",
                              "effort_requested": "medium", "elapsed_returncode": result.returncode,
                              "captured_at_utc": datetime.now(timezone.utc).isoformat()}
            try:
                envelope = json.loads(result.stdout or "{}")
                metadata["model_usage"] = envelope.get("modelUsage")
                metadata["usage"] = envelope.get("usage")
                metadata["total_cost_usd"] = envelope.get("total_cost_usd")
                metadata["model_reported"] = list((envelope.get("modelUsage") or {}).keys())
            except (ValueError, TypeError):
                metadata["usage_parse_error"] = True
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with _ledger_lock, LEDGER.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n")
        elif isinstance(command, list) and command and str(command[0]).endswith("codex") and "--json" in command:
            events = []
            for line in (result.stdout or "").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if any(k in event for k in ("usage", "token_count", "total_token_usage", "model_usage")):
                    events.append({k: event[k] for k in ("type", "usage", "token_count", "total_token_usage", "model_usage") if k in event})
            if events:
                LEDGER.parent.mkdir(parents=True, exist_ok=True)
                row = {"provider": "codex", "requested_model": "gpt-6-luna", "effort_requested": "medium",
                       "captured_at_utc": datetime.now(timezone.utc).isoformat(), "usage_events": events}
                with _ledger_lock, LEDGER.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        return result
    model_cli.subprocess.run = audited_run


def _config(model_key: str):
    if model_key == "luna":
        return "codex", "gpt-6-luna"
    if model_key == "opus":
        return "claude", "claude-opus-5-5"
    raise ValueError(model_key)


def _trajectory(model_key: str, site: str, budget: int, seed: int, phase: str):
    provider, model = _config(model_key)
    method = "pilot"  # This explicit supplemental method bypasses the frozen model allowlist only.
    state = run_trajectory(provider=provider, model=model, held_site=site, budget=budget, seed=seed,
                           method=method, output_dir=OUT / model_key)
    if state.get("budget_used") != budget or not state.get("robust_selected_quartet_key"):
        raise RuntimeError("Trajectory did not complete under the frozen source-only controller protocol")
    if state.get("agent_choice_valid") is not True:
        raise RuntimeError("Trajectory has no valid model final choice")
    state["supplemental_phase"] = phase
    state["requested_reasoning_effort"] = "medium"
    state["provider_cli_effort_flag"] = "--effort medium" if provider == "claude" else "Codex reasoning_effort=medium"
    path = OUT / model_key / method / provider / site / f"B{budget}_seed{seed}.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"model": model_key, "phase": phase, "site": site, "budget": budget, "seed": seed,
            "state_path": str(path), "budget_used": budget,
            "final_choice": state.get("agent_choice_quartet_key"), "robust_choice": state.get("robust_selected_quartet_key")}


def _zero_shot(model_key: str, site: str, seed: int):
    provider, model = _config(model_key)
    path = OUT / model_key / "zero_shot" / site / f"seed{seed}.json"
    catalog = pd.read_csv(CATALOG, dtype=str)
    if path.exists():
        # Preserve previously valid baseline picks; resume only missing cells.
        prior = json.loads(path.read_text(encoding="utf-8"))
        prior_key = str(prior.get("quartet_key", ""))
        prior_ids = prior_key.split("|")
        historical = set(pd.read_csv(PACKAGE / "data/historical_quartets.csv", dtype=str).quartet_key.astype(str))
        if (prior.get("condition") == "zero_shot" and len(prior_ids) == 4 and
                len(set(prior_ids)) == 4 and not set(prior_ids).difference(set(catalog.head_id)) and
                prior_key not in historical and prior.get("held_site") == site and int(prior.get("seed", -1)) == seed):
            return {"model": model_key, "phase": "zero_shot", "site": site, "seed": seed,
                    "quartet_key": prior_key, "resumed_existing_valid_pick": True}
    catalog_json = catalog[["head_id", "head_name", "taxonomy_category"]].to_dict("records")
    schema = {"type": "object", "required": ["final_choice", "decision_summary"],
              "additionalProperties": False,
              "properties": {"final_choice": {"type": "array", "minItems": 4, "maxItems": 4,
                                                  "items": {"type": "string"}},
                             "decision_summary": {"type": "string", "maxLength": 1000}}}
    system = PROMPT.read_text(encoding="utf-8")
    prompt = ("Zero-shot baseline: pick one quartet of four distinct catalog heads using only the catalog below. "
              "You receive no source scores, no held-site identity, and no candidate feedback. This is one pick; "
              "do not ask for data or propose alternatives. Return exactly four catalog IDs as final_choice.\n\n"
              "Catalog (head ID, name, pre-existing taxonomy category):\n" +
              json.dumps(catalog_json, ensure_ascii=False, separators=(",", ":")))
    call = model_cli.call_static(provider, model, system, prompt, schema, seed=seed,
                                 reasoning_effort="medium" if provider == "codex" else None)
    ids = sorted(map(str, call["answer"].get("final_choice", [])))
    if len(ids) != 4 or len(set(ids)) != 4 or set(ids).difference(set(catalog.head_id)):
        raise ValueError(f"Zero-shot quartet invalid for {model_key}/{site}/{seed}: {ids}")
    hist = set(pd.read_csv(PACKAGE / "data/historical_quartets.csv", dtype=str).quartet_key.astype(str))
    quartet_key = "|".join(ids)
    if quartet_key in hist:
        raise ValueError(f"Zero-shot model repeated excluded historical quartet {quartet_key}")
    source = evaluate_source(site, ids, cache_path=PACKAGE / "cache/source.sqlite")
    row = {"condition": "zero_shot", "model": model_key, "provider": provider,
           "model_requested": model, "model_reported": call["metadata"].get("model_reported"),
           "reasoning_effort_requested": "medium", "held_site": site, "seed": seed,
           "quartet_key": quartet_key, "head_ids": ids, **source,
           "decision_summary": call["answer"].get("decision_summary", ""),
           "provider_metadata": call["metadata"], "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"model": model_key, "phase": "zero_shot", "site": site, "seed": seed, "quartet_key": quartet_key}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("luna", "opus"), required=True)
    parser.add_argument("--phase", choices=("pilot", "main", "zero-shot"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.model == "opus":
        _patch_claude_effort()
        # Use the same controller-mediated local source tools as the Luna pilot.
        # The direct Claude MCP path timed out during the pilot; this keeps the
        # prompt, evaluator, validator, repair/fallback policy, and target-blind
        # information set identical without changing any frozen project file.
        trajectories.call_model = _claude_controller_round
    if args.phase == "pilot":
        jobs = [("trajectory", "I0002", 10, 9900), ("trajectory", "I0006", 50, 9901)]
    elif args.phase == "main":
        # Start long trajectories first so the complete grid has the best chance
        # of finishing within the requested wall-clock window.
        jobs = [("trajectory", site, budget, seed) for budget in (50, 25, 10)
                for site in SITES for seed in SEEDS]
    else:
        jobs = [("zero_shot", site, 0, seed) for site in SITES for seed in SEEDS]
    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {}
        for kind, site, budget, seed in jobs:
            fn = _zero_shot if kind == "zero_shot" else _trajectory
            call = (pool.submit(fn, args.model, site, seed) if kind == "zero_shot"
                    else pool.submit(fn, args.model, site, budget, seed, args.phase))
            futures[call] = (kind, site, budget, seed)
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                records.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
            except Exception as exc:
                err = {"model": args.model, "phase": args.phase, "job": key,
                       "error": f"{type(exc).__name__}: {str(exc)[:2000]}",
                       "failed_at_utc": datetime.now(timezone.utc).isoformat()}
                failures.append(err)
                fail_path = OUT / "failures.jsonl"
                fail_path.parent.mkdir(parents=True, exist_ok=True)
                with _ledger_lock, fail_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(err, sort_keys=True) + "\n")
                print("FAILED " + json.dumps(err, sort_keys=True), flush=True)
    summary = {"model": args.model, "phase": args.phase, "requested_jobs": len(jobs),
               "completed_jobs": len(records), "failed_jobs": len(failures),
               "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    summary_path = OUT / f"{args.model}_{args.phase.replace('-', '_')}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
