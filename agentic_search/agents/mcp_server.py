"""Minimal stdio MCP server exposing only the frozen search tools."""
from __future__ import annotations

import json
import math
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agentic_search.evaluator.source_evaluator import evaluate_source


PACKAGE = Path(__file__).resolve().parents[1]
STATE_PATH = Path(os.environ["AGENTIC_STATE_PATH"])
OPAQUE = os.environ.get("AGENTIC_OPAQUE_MODE", "0") == "1"
HELD_SITE = os.environ["AGENTIC_HELD_SITE"]
CURRENT_ROUND = int(os.environ["AGENTIC_ROUND"])
SEED = int(os.environ["AGENTIC_SEED"])


SEARCH_SCHEMA = {
    "type": "object", "required": ["query", "k"], "additionalProperties": False,
    "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 50}},
}
EVALUATE_SCHEMA = {
    "type": "object", "required": ["candidates"], "additionalProperties": False,
    "properties": {"candidates": {"type": "array", "minItems": 5, "maxItems": 5,
                    "items": {"type": "array", "minItems": 4, "maxItems": 4,
                              "items": {"type": "string"}}}},
}
TOOLS = [
    {"name": "search_heads", "description": "Lexically search the frozen head catalog. Returns only IDs, names, and pre-existing taxonomy categories; opaque mode returns IDs only.", "inputSchema": SEARCH_SCHEMA},
    {"name": "evaluate_batch", "description": "Evaluate exactly five new four-head candidates on the two visible source institutions. Returns only source-direction age-conditioned AUROCs, their mean/minimum/directional gap, and source-selected C. One batch per round; one repair attempt is allowed.", "inputSchema": EVALUATE_SCHEMA},
    {"name": "get_history", "description": "Return this trajectory's prior candidate proposals and source-only results in chronological order, plus budget used and remaining.", "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
]


def _read_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _write_state(state: dict[str, Any]) -> None:
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def _load_catalog() -> pd.DataFrame:
    frame = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str)
    if OPAQUE:
        mapping = pd.read_csv(PACKAGE / "data/opaque_head_mapping.csv", dtype=str)
        frame = frame.merge(mapping, on="head_id", validate="one_to_one")
        return frame.set_index("opaque_id", drop=False)
    return frame.set_index("head_id", drop=False)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", text)]


def _search_heads(query: str, k: int) -> dict[str, Any]:
    catalog = _load_catalog()
    if OPAQUE:
        identifiers = catalog.index.astype(str).tolist()
        terms = _tokens(query)
        rng = np.random.default_rng(SEED + CURRENT_ROUND + len(terms))
        if terms:
            rng.shuffle(identifiers)
        selected = identifiers[:int(k)]
        return {"heads": [{"head_id": value} for value in selected], "n_results": len(selected)}
    documents = [
        _tokens(str(row.head_name) + " " + str(row.taxonomy_category))
        for row in catalog.itertuples(index=False)
    ]
    terms = _tokens(query)
    if not terms:
        raise ValueError("query must contain at least one searchable term")
    document_frequency: dict[str, int] = {}
    for doc in documents:
        for term in set(doc):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    average_length = sum(map(len, documents)) / max(1, len(documents))
    scored = []
    for index, document in enumerate(documents):
        frequencies: dict[str, int] = {}
        for term in document:
            frequencies[term] = frequencies.get(term, 0) + 1
        score = 0.0
        for term in terms:
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue
            df = document_frequency.get(term, 0)
            inverse = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = tf + 1.2 * (1.0 - 0.75 + 0.75 * len(document) / max(average_length, 1e-9))
            score += inverse * tf * 2.2 / denominator
        scored.append((score, str(catalog.iloc[index].head_id), index))
    scored.sort(key=lambda row: (-row[0], row[1]))
    hits = []
    for score, _, index in scored[:int(k)]:
        row = catalog.iloc[index]
        hits.append({"head_id": str(row.head_id), "head_name": str(row.head_name),
                     "taxonomy_category": str(row.taxonomy_category), "lexical_score": float(score)})
    return {"heads": hits, "n_results": len(hits)}


def _history(state: dict[str, Any]) -> dict[str, Any]:
    history = sorted(state.get("history", []), key=lambda entry: (int(entry["round"]), int(entry["batch_index"])))
    if OPAQUE:
        history = [{key: value for key, value in row.items()
                    if key not in {"real_quartet_key", "real_head_ids"}}
                   for row in history]
    return {"budget": int(state["budget"]), "budget_used": int(state["budget_used"]),
            "budget_remaining": int(state["budget"] - state["budget_used"]), "history": history,
            "validation_feedback": state.get("invalid_attempts", [])[-1:]
            if state.get("invalid_attempts", []) and int(state["invalid_attempts"][-1]["round"]) == CURRENT_ROUND else []}


def _id_maps() -> tuple[dict[str, str], set[str]]:
    catalog = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str)
    valid = set(catalog.head_id)
    if not OPAQUE:
        return {value: value for value in valid}, valid
    mapping = pd.read_csv(PACKAGE / "data/opaque_head_mapping.csv", dtype=str)
    return dict(zip(mapping.opaque_id, mapping.head_id)), set(mapping.opaque_id)


def _uniform_candidate(rng: np.random.Generator, valid_ids: list[str], forbidden: set[str], seen: set[str]) -> list[str]:
    for _ in range(100_000):
        candidate = sorted(rng.choice(valid_ids, size=4, replace=False).tolist())
        key = "|".join(candidate)
        if key not in forbidden and key not in seen:
            return candidate
    raise RuntimeError("Could not draw a valid repair quartet")


def _evaluate_batch(arguments: dict[str, Any]) -> dict[str, Any]:
    state = _read_state()
    if int(state["round"]) != CURRENT_ROUND:
        raise ValueError("Agent state round does not match the tool server round")
    if state.get("round_evaluated", False):
        raise ValueError("Only one successful evaluation batch is allowed in each round")
    candidates = arguments.get("candidates")
    errors: list[str] = []
    if not isinstance(candidates, list) or len(candidates) != 5:
        errors.append("Provide exactly five candidates")
        candidates = candidates if isinstance(candidates, list) else []
    user_to_real, valid_ids = _id_maps()
    historical = set(pd.read_csv(PACKAGE / "data/historical_quartets.csv", dtype=str).quartet_key)
    existing_user = {str(row["quartet_key"]) for row in state.get("history", [])}
    existing_real = {"|".join(sorted(user_to_real[x] for x in key.split("|"))) for key in existing_user}
    normalized: list[list[str] | None] = []
    seen_in_batch: set[str] = set()
    for i, raw in enumerate(candidates[:5]):
        reason = []
        if not isinstance(raw, list) or len(raw) != 4:
            reason.append("must contain exactly four head IDs")
            normalized.append(None)
            errors.append(f"candidate {i + 1}: " + "; ".join(reason))
            continue
        quartet = [str(value) for value in raw]
        if len(set(quartet)) != 4:
            reason.append("head IDs must be distinct")
        unknown = sorted(set(quartet).difference(valid_ids))
        if unknown:
            reason.append("unknown head IDs: " + ", ".join(unknown))
        key = "|".join(sorted(quartet))
        if key in seen_in_batch:
            reason.append("duplicate within this batch")
        if not reason and not unknown:
            real_key = "|".join(sorted(user_to_real[x] for x in quartet))
            if real_key in historical:
                reason.append("historically target-inspected quartet is forbidden")
            if real_key in existing_real:
                reason.append("already queried in this trajectory")
        if reason:
            normalized.append(None)
            errors.append(f"candidate {i + 1}: " + "; ".join(reason))
        else:
            normalized.append(sorted(quartet))
            seen_in_batch.add(key)
    if len(normalized) < 5:
        normalized.extend([None] * (5 - len(normalized)))
    if errors:
        repair_attempts = int(state.get("repair_attempts", 0))
        state.setdefault("invalid_attempts", []).append({"round": CURRENT_ROUND, "errors": errors})
        if repair_attempts < 1:
            state["repair_attempts"] = repair_attempts + 1
            _write_state(state)
            return {"valid": False, "errors": errors,
                    "repair_instruction": "Repair the invalid slots once and call evaluate_batch again with exactly five valid, new quartets."}
        rng = np.random.default_rng(int(state["seed"]) + CURRENT_ROUND * 1009)
        seen_real = set(existing_real)
        for value in normalized:
            if value is not None:
                seen_real.add("|".join(sorted(user_to_real[x] for x in value)))
        repaired = []
        for value in normalized:
            if value is not None:
                repaired.append(value)
                continue
            real = _uniform_candidate(rng, sorted(user_to_real.values()), historical, seen_real)
            seen_real.add("|".join(real))
            if OPAQUE:
                real_to_user = {value: key for key, value in user_to_real.items()}
                repaired.append(sorted(real_to_user[value] for value in real))
            else:
                repaired.append(real)
        normalized = repaired
        state["auto_repair"] = True
        auto_repaired_this_batch = True
    else:
        auto_repaired_this_batch = False
    if len(normalized) != 5 or any(value is None for value in normalized):
        raise AssertionError("Candidate validator failed to repair a five-candidate batch")
    if int(state["budget_used"]) + 5 > int(state["budget"]):
        raise ValueError("This batch would exceed the trajectory's search budget")

    metrics = []
    for index, user_quartet in enumerate(normalized, start=1):
        real_quartet = sorted(user_to_real[value] for value in user_quartet)
        result = evaluate_source(HELD_SITE, real_quartet, cache_path=PACKAGE / "cache/source.sqlite")
        row = {"round": CURRENT_ROUND, "batch_index": index,
               "quartet_key": "|".join(user_quartet), "head_ids": user_quartet,
               "real_quartet_key": "|".join(real_quartet), "real_head_ids": real_quartet,
               "auc_source1_to_source2": float(result["auc_source1_to_source2"]),
               "auc_source2_to_source1": float(result["auc_source2_to_source1"]),
               "mean_source_auc": float(result["mean_source_auc"]),
               "min_source_auc": float(result["min_source_auc"]),
               "direction_gap": float(result["direction_gap"]),
               "selected_C": float(result["selected_C"]),
               "auto_repair": auto_repaired_this_batch}
        metrics.append(row)
    state.setdefault("history", []).extend(metrics)
    state["budget_used"] = int(state["budget_used"]) + 5
    state["round_evaluated"] = True
    state["last_batch"] = metrics
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_state(state)
    return {"valid": True, "round": CURRENT_ROUND, "budget_used": state["budget_used"],
            "budget_remaining": int(state["budget"] - state["budget_used"]),
            "source_results": [
                {key: value for key, value in row.items()
                 if key not in {"real_quartet_key", "real_head_ids"}}
                for row in metrics
            ]}


def _tool_result(name: str, arguments: dict[str, Any]) -> Any:
    if name == "search_heads":
        return _search_heads(str(arguments.get("query", "")), int(arguments.get("k", 20)))
    if name == "get_history":
        return _history(_read_state())
    if name == "evaluate_batch":
        return _evaluate_batch(arguments)
    raise ValueError(f"Unknown tool: {name}")


def _handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    identifier = request.get("id")
    if method == "notifications/initialized" or (method or "").startswith("notifications/"):
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": identifier,
                "result": {"protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"),
                           "capabilities": {"tools": {}},
                           "serverInfo": {"name": "brainbodyfm-source-search", "version": "1.0.0"}}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": identifier, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": identifier, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = request.get("params", {})
        try:
            value = _tool_result(str(params.get("name", "")), params.get("arguments", {}) or {})
            return {"jsonrpc": "2.0", "id": identifier,
                    "result": {"content": [{"type": "text", "text": json.dumps(value, sort_keys=True, allow_nan=False)}],
                               "structuredContent": value, "isError": False}}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            value = {"error": f"{type(exc).__name__}: {exc}"}
            return {"jsonrpc": "2.0", "id": identifier,
                    "result": {"content": [{"type": "text", "text": json.dumps(value)}],
                               "structuredContent": value, "isError": True}}
    if identifier is not None:
        return {"jsonrpc": "2.0", "id": identifier,
                "error": {"code": -32601, "message": f"Unknown method: {method}"}}
    return None


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = _handle(json.loads(line))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
        if response is not None:
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
