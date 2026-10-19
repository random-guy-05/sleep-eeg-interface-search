"""Export the frozen cohort, 539-head catalog, folds, and exclusion set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from brainbodyfm.config import load_config
from brainbodyfm.representations import STABLE4_COLUMNS
from brainbodyfm.taxonomy import coerce_bool_series


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _canonical(ids: list[str]) -> tuple[str, ...]:
    return tuple(sorted(str(x) for x in ids))


def prepare(root: Path = ROOT, out: Path = PACKAGE) -> dict[str, object]:
    for relative in ("data/source_folds", "sealed", "runs", "cache", "configs/prompts"):
        (out / relative).mkdir(parents=True, exist_ok=True)
    config = load_config(root / "brainbodyfm/frozen/analysis_config.yaml")
    manifest = pd.read_csv(root / "brainbodyfm/cache/common_interface_manifest.csv")
    required_manifest = {"subject_id", "site", "age", "outcome", "included_primary_semantic"}
    if not required_manifest.issubset(manifest.columns):
        raise ValueError(f"Canonical manifest lacks fields: {sorted(required_manifest - set(manifest))}")
    manifest = manifest.loc[coerce_bool_series(manifest["included_primary_semantic"])].copy()
    manifest["subject_id"] = manifest["subject_id"].astype(str)
    manifest["site"] = manifest["site"].astype(str)
    manifest["age"] = pd.to_numeric(manifest["age"], errors="raise").astype(float)
    manifest["outcome"] = pd.to_numeric(manifest["outcome"], errors="raise").astype(int)
    if manifest.subject_id.duplicated().any() or len(manifest) != 338:
        raise ValueError(f"Expected 338 unique primary-interface participants, found {len(manifest)}")
    if set(manifest.site) != set(config.sites):
        raise ValueError("Manifest institutions do not match the frozen BrainBodyFM folds")

    raw = pd.read_parquet(root / "brainbodyfm/cache/physiology_semantic.parquet")
    raw["subject_id"] = raw["subject_id"].astype(str)
    taxonomy = pd.read_csv(root / "brainbodyfm/frozen/head_taxonomy.csv")
    taxonomy = taxonomy.loc[coerce_bool_series(taxonomy["include_physiology_only"])].sort_values("head_index")
    feature_names = taxonomy.head_name.astype(str).tolist()
    if len(feature_names) != 539 or len(set(feature_names)) != 539:
        raise ValueError(f"Frozen physiology taxonomy must contain 539 unique heads, got {len(feature_names)}")
    if set(feature_names) != set(raw.columns) - {"subject_id"}:
        raise ValueError("Taxonomy and cached physiology feature columns differ")
    raw = raw.set_index("subject_id").loc[manifest.subject_id, feature_names].reset_index()
    if not raw.subject_id.equals(manifest.subject_id.reset_index(drop=True)):
        raise AssertionError("Feature rows failed exact manifest alignment")

    # These IDs only provide a short, deterministic interface. The mapping is
    # fixed by the original taxonomy order and carries no outcome information.
    catalog = pd.DataFrame({
        "head_id": [f"H{i:04d}" for i in range(1, len(feature_names) + 1)],
        "head_name": feature_names,
        "taxonomy_category": taxonomy.physiological_family.fillna("").astype(str).tolist(),
        "head_index": taxonomy.head_index.astype(int).tolist(),
    })
    if (catalog.taxonomy_category == "").any():
        raise ValueError("An included physiology head lacks its frozen family category")
    catalog.to_csv(out / "data/heads.csv", index=False)
    id_to_name = dict(zip(catalog.head_id, catalog.head_name))
    name_to_id = {value: key for key, value in id_to_name.items()}

    participant_features = manifest[["subject_id", "site", "age"]].merge(raw, on="subject_id", validate="one_to_one")
    _write_parquet(participant_features, out / "data/participants.parquet")
    labels = manifest[["subject_id", "site", "age", "outcome"]].copy()
    target_labels = []
    folds: dict[str, object] = {"site_order": list(config.sites), "age_window_years": config.age_window, "folds": {}}
    for held_site in config.sites:
        source_sites = [site for site in config.sites if site != held_site]
        source_rows = manifest[manifest.site.isin(source_sites)][["subject_id", "site", "age", "outcome"]]
        source_fold = source_rows.merge(raw, on="subject_id", validate="one_to_one")
        if set(source_fold.site) != set(source_sites) or held_site in set(source_fold.site):
            raise AssertionError(f"Source fold for {held_site} contains the wrong institutions")
        _write_parquet(source_fold, out / f"data/source_folds/{held_site}.parquet")
        held_labels = labels.loc[labels.site.eq(held_site)].copy()
        held_labels.insert(0, "held_site", held_site)
        target_labels.append(held_labels)
        folds["folds"][held_site] = {"source_sites": source_sites, "held_site": held_site}
    _write_parquet(pd.concat(target_labels, ignore_index=True), out / "sealed/target_labels.parquet")
    (out / "data/folds.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")

    historical: dict[tuple[str, ...], set[str]] = {}
    panel_path = root / "brainbodyfm/results/probe_search/candidate_panel.parquet"
    panel = pd.read_parquet(panel_path)
    candidates = panel.drop_duplicates("draw").sort_values("draw")
    if len(candidates) != 500:
        raise ValueError(f"Historical random-4 panel must contain 500 candidates, found {len(candidates)}")
    for row in candidates.itertuples():
        names = json.loads(row.heads)
        if len(names) != 4 or any(name not in name_to_id for name in names):
            raise ValueError(f"Invalid historical candidate draw {row.draw}")
        key = _canonical([name_to_id[name] for name in names])
        historical.setdefault(key, set()).add("historical_random500")
    stable_key = _canonical([name_to_id[name] for name in STABLE4_COLUMNS])
    historical.setdefault(stable_key, set()).add("historical_stable4")
    records = []
    for key, sources in sorted(historical.items()):
        records.append({"quartet_key": "|".join(key), "source": ";".join(sorted(sources),),
                        **{f"head_{i + 1}": head for i, head in enumerate(key)}})
    history_frame = pd.DataFrame(records)
    history_frame.to_csv(out / "data/historical_quartets.csv", index=False)
    if len(history_frame) < 500 or len(history_frame) > 501:
        raise ValueError(f"Expected 500 historical candidates plus at most Stable4; got {len(history_frame)}")

    summary = {
        "n_participants": len(manifest),
        "n_heads": len(catalog),
        "n_historical_quartets": len(history_frame),
        "n_stable4_overlap_with_random500": int("historical_random500;historical_stable4" in set(history_frame.source)),
        "sites": {site: {"n": int(len(g)), "n_positive": int(g.outcome.sum())}
                  for site, g in manifest.groupby("site", sort=False)},
        "taxonomy_categories": catalog.taxonomy_category.value_counts().sort_index().to_dict(),
    }
    (out / "data/preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=PACKAGE)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
