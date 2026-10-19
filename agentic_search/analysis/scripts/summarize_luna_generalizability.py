"""Write auditable quartet, overlap, invalid-event, and call summaries for Luna."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "analysis/output"
ROOT = PACKAGE / "runs/supplemental/generalizability/luna"
SITES = ("I0002", "I0006", "S0001")
BUDGETS = (10, 25, 50)


def _ids(value) -> list[str]:
    if isinstance(value, list):
        return sorted(map(str, value))
    return sorted(str(value).split("|"))


def main() -> None:
    catalog = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str).set_index("head_id")
    sol = pd.read_csv(OUT / "agent_choice_secondary_per_trajectory.csv")
    sol = sol.rename(columns={"agent_choice_quartet_key": "sol_quartet_key"})
    sol["sol_ids"] = sol.sol_quartet_key.map(_ids)
    sol_lookup = {(str(r.held_site), int(r.budget), int(r.seed)): r.sol_ids
                  for r in sol.itertuples(index=False)}

    luna = pd.read_csv(OUT / "luna_final_choice_scored.csv")
    rows = []
    for r in luna.itertuples(index=False):
        ids = _ids(r.quartet_key)
        categories = [str(catalog.loc[head, "taxonomy_category"]) for head in ids]
        unique_categories = sorted(set(categories))
        sol_ids = sol_lookup.get((str(r.held_site), int(r.budget), int(r.seed)), [])
        shared = sorted(set(ids) & set(sol_ids))
        union = set(ids) | set(sol_ids)
        rows.append({"controller": "GPT-6 Luna", "condition": "adaptive_final_choice",
                     "held_site": str(r.held_site), "budget": int(r.budget), "seed": int(r.seed),
                     "quartet_key": str(r.quartet_key), "target_auc": float(r.target_auc),
                     "head_ids": "|".join(ids), "head_categories": " | ".join(categories),
                     "unique_categories": " | ".join(unique_categories),
                     "n_distinct_categories": len(unique_categories),
                     "sol_matched": bool(sol_ids),
                     "sol_exact_quartet_overlap": bool(sol_ids and ids == sol_ids),
                     "sol_shared_head_count": len(shared),
                     "sol_jaccard": len(shared) / len(union) if union else float("nan")})

    robust_scores = pd.read_csv(OUT / "luna_generalizability_robust_choices.csv")
    robust_target = pd.read_parquet(PACKAGE / "runs/supplemental/held_site/generalizability_target_results.parquet")
    for r in robust_scores.itertuples(index=False):
        ids = _ids(r.quartet_key)
        categories = [str(catalog.loc[head, "taxonomy_category"]) for head in ids]
        sol_ids = sol_lookup.get((str(r.held_site), int(r.budget), int(r.seed)), [])
        shared = sorted(set(ids) & set(sol_ids))
        union = set(ids) | set(sol_ids)
        robust_score = robust_target[(robust_target.held_site.astype(str) == str(r.held_site)) &
                                     (robust_target.quartet_key.astype(str) == str(r.quartet_key))]
        if robust_score.empty:
            raise ValueError(f"Luna robust selection lacks a held-site score: {r.held_site}/{r.quartet_key}")
        rows.append({"controller": "GPT-6 Luna", "condition": "adaptive_source_robust",
                     "held_site": str(r.held_site), "budget": int(r.budget), "seed": int(r.seed),
                     "quartet_key": str(r.quartet_key), "target_auc": float(robust_score.iloc[0].target_auc),
                     "head_ids": "|".join(ids), "head_categories": " | ".join(categories),
                     "unique_categories": " | ".join(sorted(set(categories))),
                     "n_distinct_categories": len(set(categories)), "sol_matched": bool(sol_ids),
                     "sol_exact_quartet_overlap": bool(sol_ids and ids == sol_ids),
                     "sol_shared_head_count": len(shared),
                     "sol_jaccard": len(shared) / len(union) if union else float("nan")})

    zero = []
    zero_scores = pd.read_csv(OUT / "luna_zero_shot_scored.csv")
    for path in sorted((ROOT / "zero_shot").glob("*/*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if state.get("condition") != "zero_shot":
            continue
        site = str(state["held_site"])
        seed = int(state["seed"])
        ids = _ids(state["head_ids"])
        key = "|".join(ids)
        score_rows = zero_scores[(zero_scores.held_site.astype(str) == site) &
                                 (zero_scores.quartet_key.astype(str) == key)]
        if score_rows.empty:
            raise ValueError(f"Zero-shot Luna pick lacks a held-site score: {site}/{key}")
        categories = [str(catalog.loc[head, "taxonomy_category"]) for head in ids]
        sol_ids = sol_lookup.get((site, 10, seed), [])
        shared = sorted(set(ids) & set(sol_ids))
        union = set(ids) | set(sol_ids)
        zero.append({"controller": "GPT-6 Luna", "condition": "zero_shot",
                     "held_site": site, "budget": 0, "seed": seed,
                     "quartet_key": key, "target_auc": float(score_rows.iloc[0].target_auc),
                     "head_ids": "|".join(ids), "head_categories": " | ".join(categories),
                     "unique_categories": " | ".join(sorted(set(categories))),
                     "n_distinct_categories": len(set(categories)), "sol_matched": bool(sol_ids),
                     "sol_exact_quartet_overlap": bool(sol_ids and ids == sol_ids),
                     "sol_shared_head_count": len(shared),
                     "sol_jaccard": len(shared) / len(union) if union else float("nan")})
    details = pd.DataFrame(rows + zero)
    details.to_csv(OUT / "luna_generalizability_quartet_details.csv", index=False)

    states = []
    invalid_rows = []
    for path in sorted((ROOT / "pilot/codex").glob("**/*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if int(state.get("seed", -1)) not in range(1000, 1010) or int(state.get("budget", 0)) not in BUDGETS:
            continue
        states.append((path, state))
        for event_index, event in enumerate(state.get("invalid_attempts", []), 1):
            invalid_rows.append({"held_site": state.get("held_site_internal"),
                                 "budget": int(state["budget"]), "seed": int(state["seed"]),
                                 "round": event.get("round"), "event_index": event_index,
                                 "validation_errors": "; ".join(map(str, event.get("errors", []))),
                                 "rejected_quartet_contents_persisted": False,
                                 "candidate_content_note": "Original controller response for rejected batch was not retained in this run artifact."})
    invalid = pd.DataFrame(invalid_rows)
    invalid.to_csv(OUT / "luna_generalizability_invalid_diagnostics.csv", index=False)

    state_by_cell = {(str(s["held_site_internal"]), int(s["budget"]), int(s["seed"])): s
                     for _, s in states}
    call_count = Counter()
    durations = Counter()
    repair_count = 0
    log_root = ROOT / "logs"
    expected_rounds = 0
    for site in SITES:
        for budget in BUDGETS:
            for seed in range(1000, 1010):
                state = state_by_cell[(site, budget, seed)]
                expected_rounds += budget // 5
                log_path = log_root / f"pilot_codex_{site}_B{budget}_seed{seed}.jsonl"
                if not log_path.is_file():
                    continue
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    for outer in event.get("provider_calls", []):
                        metadata = outer.get("metadata", {})
                        proposals = metadata.get("controller_proposal_calls", [])
                        call_count["proposal_calls"] += len(proposals)
                        call_count["final_choice_calls"] += int(metadata.get("final_choice_call") is not None)
                        repair_count += int(metadata.get("controller_repair_attempts", 0))
                        for proposal in proposals:
                            if proposal.get("elapsed_seconds") is not None:
                                durations["proposal_seconds"] += float(proposal["elapsed_seconds"])
                                durations["proposal_timed_calls"] += 1
                        choice = metadata.get("final_choice_call") or {}
                        if choice.get("elapsed_seconds") is not None:
                            durations["choice_seconds"] += float(choice["elapsed_seconds"])
                            durations["choice_timed_calls"] += 1

    complete = sum(len(s.get("history", [])) == int(s.get("budget", -1)) for _, s in states)
    valid = sum(s.get("agent_choice_valid") is True for _, s in states)
    invalid_finals = [{"held_site": s.get("held_site_internal"), "budget": int(s["budget"]),
                       "seed": int(s["seed"]), "error": s.get("agent_choice_error")}
                      for _, s in states if s.get("agent_choice_valid") is not True]
    invalid_hist = Counter()
    for row in invalid_rows:
        invalid_hist[row["validation_errors"]] += 1
    usage = []
    ledger = ROOT.parent / "provider_usage.jsonl"
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("requested_model") == "gpt-6-luna":
                usage.append(row)
    invalid_batch_rate = (len(invalid_rows) / call_count["proposal_calls"]
                          if call_count["proposal_calls"] else None)
    summary = {
        "controller": "GPT-6 Luna", "effort": "medium", "temperature_controlled": False,
        "adaptive_grid": {"expected_trajectories": 90, "source_histories_complete": complete,
                          "valid_final_choices": valid, "invalid_final_choices": invalid_finals,
                          "expected_successful_source_batches": expected_rounds},
        "zero_shot_valid_picks": len(zero),
        "invalid_evaluator_events": len(invalid_rows),
        "invalid_evaluator_batch_rejection_rate": invalid_batch_rate,
        "invalid_evaluator_event_note": "Rejected batch response contents were not persisted, so the exact invalid-quartet-slot rate cannot be reconstructed; the batch rejection rate is event count divided by logged model proposal calls.",
        "validation_error_frequency": dict(invalid_hist),
        "provider_calls_from_retained_success_logs": dict(call_count),
        "repair_calls_recorded": repair_count,
        "proposal_elapsed_seconds_total": round(durations["proposal_seconds"], 3),
        "proposal_elapsed_timed_call_count": durations["proposal_timed_calls"],
        "final_choice_elapsed_seconds_total": round(durations["choice_seconds"], 3),
        "final_choice_elapsed_timed_call_count": durations["choice_timed_calls"],
        "token_or_billing_records_for_luna": len(usage),
        "monetary_cost_usd": None,
        "cost_note": "No Luna token or billing usage rows were retained; cost is unavailable, not zero.",
        "manifest_and_scoring_note": "See separately frozen generalizability manifest/release and Luna score artifacts.",
    }
    (OUT / "luna_generalizability_controller_accounting.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    def summarize(frame: pd.DataFrame, condition: str) -> list[dict]:
        output = []
        for budget, part in frame[frame.condition == condition].groupby("budget", sort=True):
            output.append({"condition": condition, "budget": int(budget), "n": len(part),
                           "mean_categories": float(part.n_distinct_categories.mean()),
                           "exact_sol_overlap_rate": float(part.sol_exact_quartet_overlap.mean()),
                           "mean_shared_heads_with_sol": float(part.sol_shared_head_count.mean()),
                           "mean_jaccard_with_sol": float(part.sol_jaccard.mean())})
        return output
    summary["category_overlap_summary"] = (summarize(details, "adaptive_final_choice") +
                                            summarize(details, "adaptive_source_robust") +
                                            summarize(details, "zero_shot"))
    (OUT / "luna_generalizability_controller_accounting.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
