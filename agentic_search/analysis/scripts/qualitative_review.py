"""§30 qualitative review: print the three predeclared trajectories with prose from round logs."""
import glob
import json

import pandas as pd

review = json.load(open("agentic_search/analysis/output/manual_trajectory_review.json"))
for entry in review:
    eid = entry["experiment_id"]
    print("=" * 100)
    print(f"[{entry['selection'].upper()}] {eid}  site={entry['held_site']} B={entry['budget']} "
          f"seed={entry['seed']}  selected target AUC={entry['robust_selected_target_auc']:.4f}")
    print(f"rank rule: {entry['rank_rule']}")
    ev = pd.DataFrame(entry["events"])
    with pd.option_context("display.width", 250, "display.max_rows", 60,
                           "display.max_colwidth", 42):
        ev2 = ev.copy()
        ev2["cats"] = ev2["taxonomy_categories"].apply(lambda c: ",".join(x[:14] for x in c))
        print(ev2[["round", "batch_index", "mean_source_auc", "min_source_auc",
                   "target_auc_posthoc", "cats"]].to_string(index=False))
    print("--- agent prose (decision_summary per round) ---")
    parts = eid.split("__")
    provider = parts[1].split("-")[0]
    log = (f"agentic_search/runs/source_only/agents/logs/"
           f"{parts[0]}_{provider}_{parts[2]}_{parts[3]}_{parts[4]}.jsonl")
    for line in open(log):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        summary = rec.get("decision_summary", "")
        proposed = rec.get("proposed_candidates", [])
        print(f"\n[round {rec.get('round')}] proposed {len(proposed)} quartets | "
              f"invalid_attempts={rec.get('invalid_attempts')} auto_repair={rec.get('auto_repair')}")
        print(f"  {summary}")
    print()
