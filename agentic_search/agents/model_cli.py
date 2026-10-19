"""Isolated provider-CLI calls for source-only agent turns."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "agentic_search"
PYTHON = ROOT / ".venv/bin/python"
MCP_SERVER = PACKAGE / "agents/mcp_server.py"


def _tool_env(state_path: Path, held_site: str, round_index: int, seed: int, budget: int, opaque: bool) -> dict[str, str]:
    return {
        "AGENTIC_STATE_PATH": str(state_path),
        "AGENTIC_HELD_SITE": str(held_site),
        "AGENTIC_ROUND": str(round_index),
        "AGENTIC_SEED": str(seed),
        "AGENTIC_BUDGET": str(budget),
        "AGENTIC_OPAQUE_MODE": "1" if opaque else "0",
        "PYTHONPATH": str(ROOT),
    }


def _response_schema(final_round: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "decision_summary": {"type": "string", "maxLength": 1000},
        "candidates": {"type": "array", "minItems": 5, "maxItems": 5,
                        "items": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "string"}}},
    }
    required = ["decision_summary", "candidates"]
    if final_round:
        properties["final_choice"] = {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "string"}}
        required.append("final_choice")
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model did not return a JSON object")
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def _mcp_exchange(requests: list[dict[str, Any]], env: dict[str, str]) -> list[dict[str, Any]]:
    lines = [json.dumps(row, separators=(",", ":")) for row in requests]
    proc = subprocess.run([str(PYTHON), str(MCP_SERVER)], input="\n".join(lines) + "\n",
                          text=True, capture_output=True, env=env, timeout=120, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Local source-tool server failed: {proc.stderr[-1200:]}")
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def _mcp_tool_catalog(env: dict[str, str]) -> list[dict[str, Any]]:
    responses = _mcp_exchange([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "agentic-search-runner", "version": "1.0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ], env)
    response = next((row for row in responses if row.get("id") == 2), {})
    if "error" in response:
        raise RuntimeError(f"Could not load the frozen source-tool catalog: {response['error']}")
    return list(response.get("result", {}).get("tools", []))


def _mcp_tool_call(name: str, arguments: dict[str, Any], env: dict[str, str]) -> Any:
    responses = _mcp_exchange([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "agentic-search-runner", "version": "1.0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": name, "arguments": arguments}},
    ], env)
    response = next((row for row in responses if row.get("id") == 2), {})
    if "error" in response:
        raise RuntimeError(f"Local source-tool call failed: {response['error']}")
    result = response.get("result", {})
    if "structuredContent" in result:
        return result["structuredContent"]
    content = result.get("content", [])
    if content and content[0].get("type") == "text":
        return json.loads(content[0]["text"])
    return result


def _ollama_call(messages: list[dict[str, Any]], *, model: str, schema: dict[str, Any],
                 temperature: float, seed: int, tools: list[dict[str, Any]] | None = None,
                 tool_env: dict[str, str] | None = None,
                 max_steps: int = 8) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use a local Ollama endpoint with a model ID, structured output, and fixed sampling."""
    request_payload: dict[str, Any] = {
        "model": model, "messages": messages, "stream": False, "format": schema,
        "options": {"temperature": float(temperature), "seed": int(seed), "num_predict": 4096},
    }
    if tools:
        request_payload["tools"] = tools
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    tool_names: list[str] = []
    tool_counts: dict[str, int] = {}
    evaluated = False
    if tools and tool_env:
        try:
            existing_state = json.loads(Path(tool_env["AGENTIC_STATE_PATH"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, KeyError):
            existing_state = {}
        if existing_state.get("round_evaluated"):
            recovery_history = _mcp_tool_call("get_history", {}, tool_env)
            messages.append({"role": "user", "content":
                "The source-evaluation batch already succeeded before an interrupted model response. "
                "Here is the full current history: " +
                json.dumps(recovery_history, separators=(",", ":"), allow_nan=False) +
                ". Return the required JSON now, echoing the five most recent batch candidates in order. "
                "On the final round, choose one evaluated quartet as final_choice."})
            request_payload.pop("tools", None)
            evaluated = True
    for _ in range(max_steps):
        request = Request("http://127.0.0.1:11434/api/chat",
                          data=json.dumps(request_payload, separators=(",", ":")).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        for retry in range(4):
            try:
                with urlopen(request, timeout=900) as response:
                    envelope = json.load(response)
                break
            except HTTPError as exc:
                if exc.code not in {408, 425, 429, 500, 502, 503, 504} or retry == 3:
                    raise RuntimeError(f"Ollama request failed (HTTP {exc.code})") from exc
                time.sleep(min(2 ** retry, 8))
                usage["http_retries"] = usage.get("http_retries", 0) + 1
            except Exception as exc:
                if retry == 3:
                    raise RuntimeError(f"Ollama request failed ({type(exc).__name__})") from exc
                time.sleep(min(2 ** retry, 8))
                usage["http_retries"] = usage.get("http_retries", 0) + 1
        usage["requests"] += 1
        usage["prompt_tokens"] += int(envelope.get("prompt_eval_count") or 0)
        usage["completion_tokens"] += int(envelope.get("eval_count") or 0)
        if envelope.get("error"):
            raise RuntimeError(f"Ollama model error: {str(envelope['error'])[:1000]}")
        message = envelope.get("message") or {}
        calls = message.get("tool_calls") or []
        if calls:
            messages.append(message)
            if not tools:
                raise RuntimeError("Ollama returned a tool call in a tool-free semantic generation")
            for call in calls:
                function = call.get("function", {})
                name = str(function.get("name", ""))
                tool_names.append(name)
                tool_counts[name] = tool_counts.get(name, 0) + 1
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if tool_env is None:
                    raise RuntimeError("Ollama requested a tool without a scoped source-tool environment")
                tool_result = _mcp_tool_call(name, arguments or {}, tool_env)
                if name == "evaluate_batch" and isinstance(tool_result, dict) and tool_result.get("valid") is True:
                    evaluated = True
                if tool_env:
                    try:
                        current_state = json.loads(Path(tool_env["AGENTIC_STATE_PATH"]).read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        current_state = {}
                    evaluated = evaluated or bool(current_state.get("round_evaluated"))
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(tool_result, separators=(",", ":"), allow_nan=False)})
                if name == "get_history":
                    request_payload["tools"] = [tool for tool in (request_payload.get("tools") or [])
                                                 if tool["function"]["name"] != "get_history"]
                    messages.append({"role": "user", "content":
                        "You have the complete trajectory history. Propose exactly five new quartets and submit them "
                        "using evaluate_batch. Do not call get_history again."})
                elif name == "search_heads" and tool_counts[name] >= 3:
                    request_payload["tools"] = [tool for tool in (request_payload.get("tools") or [])
                                                 if tool["function"]["name"] != "search_heads"]
                    messages.append({"role": "user", "content":
                        "Catalog lookup is complete for this round. Submit five new quartets using evaluate_batch."})
            if evaluated:
                request_payload.pop("tools", None)
                messages.append({"role": "user", "content":
                    "Your source-evaluation batch has now completed successfully. No more tools are available "
                    "for this round. Return the required JSON now. Echo exactly the five evaluated candidates "
                    "in the same order. On the final round, also choose one evaluated quartet as final_choice."})
            continue
        parsed = _parse_json(str(message.get("content", "")))
        usage["model_reported"] = envelope.get("model")
        usage["temperature"] = float(temperature)
        usage["sampling_seed"] = int(seed)
        usage["done_reason"] = envelope.get("done_reason")
        return parsed, usage
    raise RuntimeError(f"Ollama did not finish after {max_steps} response/tool steps; "
                       f"tools={tool_names}; evaluated={evaluated}")


def _ollama_digest(model: str) -> str | None:
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
            tags = json.load(response).get("models", [])
        return next((str(row.get("digest")) for row in tags if row.get("name") == model), None)
    except Exception:
        return None


def call_static(provider: str, model: str, system_prompt: str, user_prompt: str,
                schema: dict[str, Any], *, timeout_seconds: int = 600,
                seed: int = 2026092401,
                reasoning_effort: str | None = None) -> dict[str, Any]:
    """Call a model with no files, browser, shell, or MCP tools."""
    schema_json = json.dumps(schema, separators=(",", ":"))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="brainbodyfm-static-") as workdir_raw:
        workdir = Path(workdir_raw)
        if provider == "codex":
            schema_path = workdir / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            output_path = workdir / "last-message.json"
            command = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                       "--sandbox", "read-only", "--model", model]
            if reasoning_effort:
                command.extend(["--config", f"model_reasoning_effort={json.dumps(reasoning_effort)}"])
            command.extend([
                       "--cd", str(workdir), "--skip-git-repo-check", "--json",
                       "--output-schema", str(schema_path), "--output-last-message", str(output_path),
                       "--disable", "shell_tool", "--disable", "unified_exec", "--disable", "unified_exec_tty",
                       "--disable", "browser_use", "--disable", "browser_use_external",
                       "--disable", "computer_use", "--disable", "multi_agent",
                       "--disable", "plugins", "--disable", "in_app_browser", "--disable", "in_app_chat", "-"])
            full_prompt = system_prompt + "\n\n--- Request ---\n" + user_prompt
            proc = subprocess.run(command, input=full_prompt, text=True, capture_output=True,
                                  cwd=workdir, env=os.environ.copy(), timeout=timeout_seconds, check=False)
            response = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            if proc.returncode != 0 or not response.strip():
                raise RuntimeError(f"Codex CLI failed (exit {proc.returncode}): {proc.stderr[-1500:]}")
            parsed = _parse_json(response)
            model_reported = model
        elif provider == "claude":
            command = ["claude", "--print", "--model", model, "--output-format", "json", "--tools", "",
                       "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence", "--restricted",
                       "--json-schema", schema_json, "--system-prompt", system_prompt, user_prompt]
            proc = subprocess.run(command, text=True, capture_output=True, cwd=workdir,
                                  env=os.environ.copy(), timeout=timeout_seconds, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"Claude CLI failed (exit {proc.returncode}): {proc.stderr[-1500:]}")
            envelope = json.loads(proc.stdout)
            parsed = _parse_json(str(envelope.get("result", "")))
            usage = envelope.get("modelUsage") or {}
            model_reported = next(iter(usage), None)
        elif provider == "ollama":
            started_messages = [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}]
            parsed, usage = _ollama_call(started_messages, model=model, schema=schema,
                                         temperature=0.2, seed=seed)
            model_reported = usage.get("model_reported")
            return {"answer": parsed,
                    "metadata": {"provider": provider, "model_requested": model,
                                 "model_reported": model_reported, "model_digest": _ollama_digest(model),
                                 "elapsed_seconds": float(time.monotonic() - started),
                                 "temperature_controlled": True, "temperature": 0.2,
                                 "sampling_seed": int(seed), "usage": usage,
                                 "cli_version": _version_line(["ollama", "--version"]),
                                 "runtime": "ollama_local_api"}}
        else:
            raise ValueError(f"Unknown provider CLI: {provider}")
    return {"answer": parsed,
            "metadata": {"provider": provider, "model_requested": model, "model_reported": model_reported,
                         "elapsed_seconds": float(time.monotonic() - started),
                         "reasoning_effort": reasoning_effort if provider == "codex" else None,
                         "temperature_controlled": False,
                         "cli_version": _version_line([provider if provider == "claude" else "codex", "--version"])}}


def call_model(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    state_path: Path,
    held_site: str,
    round_index: int,
    seed: int,
    budget: int,
    opaque: bool,
    final_round: bool,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    schema = _response_schema(final_round)
    schema_json = json.dumps(schema, separators=(",", ":"))
    state_path = state_path.resolve()
    env = os.environ.copy()
    scoped_env = _tool_env(state_path, held_site, round_index, seed, budget, opaque)
    env.update(scoped_env)
    started = time.monotonic()
    complete_user_prompt = system_prompt + "\n\n--- Round instructions ---\n" + user_prompt
    with tempfile.TemporaryDirectory(prefix="brainbodyfm-agent-") as workdir_raw:
        workdir = Path(workdir_raw)
        if provider == "codex":
            codex_home = workdir / "codex-home"
            codex_home.mkdir()
            auth_source = Path.home() / ".codex/auth.json"
            if not auth_source.is_file():
                raise RuntimeError("Codex CLI authentication file is unavailable")
            shutil.copyfile(auth_source, codex_home / "auth.json")
            (codex_home / "auth.json").chmod(0o600)
            env["CODEX_HOME"] = str(codex_home)
            mcp_env = ",".join(f"{json.dumps(k)}={json.dumps(v)}" for k, v in scoped_env.items())
            server_config = (f'mcp_servers.agent_tools={{command={json.dumps(str(PYTHON))},'
                             f'args=[{json.dumps(str(MCP_SERVER))}],env={{{mcp_env}}}}}')
            output_path = workdir / "last-message.json"
            command = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                       "--approve-for-me", "--model", model, "--cd", str(workdir), "--skip-git-repo-check",
                       "--json", "--config", server_config, "--output-schema", str(workdir / "schema.json"),
                       "--output-last-message", str(output_path), "--disable", "shell_tool",
                       "--disable", "unified_exec", "--disable", "unified_exec_tty",
                       "--disable", "browser_use", "--disable", "browser_use_external",
                       "--disable", "computer_use", "--disable", "multi_agent",
                       "--enable", "mcp_2026_07_28", "--disable", "plugins", "--disable", "in_app_browser",
                       "--disable", "in_app_chat", "-"]
            (workdir / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
            proc = subprocess.run(command, input=complete_user_prompt, text=True, capture_output=True,
                                  cwd=workdir, env=env, timeout=timeout_seconds, check=False)
            response_text = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            if proc.returncode != 0 or not response_text.strip():
                raise RuntimeError(f"Codex CLI failed (exit {proc.returncode}): {proc.stderr[-1500:]}")
            parsed = _parse_json(response_text)
            event_types, tool_events = [], []
            for line in proc.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_types.append(str(event.get("type", "unknown")))
                item = event.get("item") or event.get("data", {}).get("item") or {}
                if "tool" in str(event.get("type", "")).casefold() or "tool" in str(item.get("type", "")).casefold():
                    tool_events.append({"event_type": event.get("type"), "item_type": item.get("type"),
                                        "name": item.get("name") or item.get("tool_name")})
            metadata = {"provider": "codex", "model_requested": model,
                        "model_reported": model,
                        "cli_version": _version_line(["codex", "--version"]),
                        "mcp_tool_approval": "auto_review",
                        "temperature_controlled": False,
                        "cli_event_types": event_types,
                        "cli_tool_events": tool_events}
        elif provider == "claude":
            server_config_path = workdir / "mcp-config.json"
            server_config_path.write_text(json.dumps({"mcpServers": {"agent_tools": {
                "command": str(PYTHON), "args": [str(MCP_SERVER)], "env": scoped_env}}}), encoding="utf-8")
            command = ["claude", "--print", "--model", model, "--output-format", "json",
                       "--tools", "", "--strict-mcp-config", "--mcp-config", str(server_config_path),
                       "--allowedTools",
                       "mcp__agent_tools__search_heads,mcp__agent_tools__get_history,mcp__agent_tools__evaluate_batch",
                       "--setting-sources", "", "--no-session-persistence", "--restricted",
                       "--json-schema", schema_json, "--system-prompt", system_prompt,
                       user_prompt]
            proc = subprocess.run(command, text=True, capture_output=True, cwd=workdir,
                                  env=env, timeout=timeout_seconds, check=False)
            if proc.returncode != 0:
                diagnostic = (proc.stderr or proc.stdout)[-3000:]
                raise RuntimeError(f"Claude CLI failed (exit {proc.returncode}): {diagnostic}")
            envelope = json.loads(proc.stdout)
            response_text = str(envelope.get("result", ""))
            parsed = _parse_json(response_text)
            usage = envelope.get("modelUsage") or {}
            reported_model = next(iter(usage), None)
            metadata = {"provider": "claude", "model_requested": model,
                        "model_reported": reported_model,
                        "cli_version": _version_line(["claude", "--version"]),
                        "mcp_tool_approval": "explicit_allowlist_only",
                        "temperature_controlled": False}
        elif provider == "ollama":
            tool_catalog = _mcp_tool_catalog(env)
            ollama_tools = [{"type": "function", "function": {
                "name": tool["name"], "description": tool["description"],
                "parameters": tool["inputSchema"]}}
                for tool in tool_catalog]
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}]
            ollama_seed = int(seed) + int(round_index) * 100_003
            parsed, usage = _ollama_call(messages, model=model, schema=schema,
                                         temperature=0.2, seed=ollama_seed,
                                         tools=ollama_tools, tool_env=env, max_steps=8)
            metadata = {"provider": "ollama", "model_requested": model,
                        "model_reported": usage.get("model_reported"),
                        "model_digest": _ollama_digest(model),
                        "cli_version": _version_line(["ollama", "--version"]),
                        "runtime": "ollama_local_api", "temperature": 0.2,
                        "temperature_controlled": True,
                        "sampling_seed": ollama_seed, "usage": usage,
                        "mcp_tool_approval": "locally_scoped_source_tools"}
            proc = None
        else:
            raise ValueError(f"Unknown provider CLI: {provider}")
    metadata["elapsed_seconds"] = float(time.monotonic() - started)
    summary = str(parsed.get("decision_summary", ""))
    if len(summary.split()) > 80:
        raise ValueError("decision_summary exceeded the frozen 80-word limit")
    if not isinstance(parsed.get("candidates"), list) or len(parsed["candidates"]) != 5:
        raise ValueError("Agent response must echo exactly five candidates")
    return {"answer": parsed, "metadata": metadata,
            "cli_stderr_tail": proc.stderr[-500:] if proc and proc.stderr else ""}


def _version_line(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
        return (result.stdout or result.stderr).strip().splitlines()[0]
    except Exception:
        return "unavailable"
