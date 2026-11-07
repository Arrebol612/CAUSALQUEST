"""Strict parsing from common model tool formats to canonical harness actions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from causalquest.harness.contracts import AgentAction

STATE_TOOL_NAMES = {"cq_update_investigation_state", "update_investigation_state"}
REPORT_TOOL_NAMES = {"cq_submit_review", "submit_review"}
FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
HERMES_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return dict(value)


def _from_function(name: str, arguments: Any, call_id: str | None = None) -> AgentAction:
    args = _mapping(arguments or {}, name="tool arguments")
    if name in STATE_TOOL_NAMES:
        payload = {"kind": "state_update", "patch": args}
    elif name in REPORT_TOOL_NAMES:
        payload = {"kind": "final_report", "report": args.get("response", args)}
    else:
        payload = {"kind": "tool_call", "tool_name": name, "arguments": args}
    if call_id:
        payload["client_action_id"] = call_id
    return AgentAction.from_mapping(payload)


def _parse_tool_call(call: Mapping[str, Any]) -> AgentAction:
    function = call.get("function", call)
    if not isinstance(function, Mapping):
        raise TypeError("tool call function must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call is missing a function name")
    call_id = call.get("id")
    return _from_function(name.strip(), function.get("arguments", {}), str(call_id) if call_id else None)


def _content_objects(content: str) -> list[dict[str, Any]]:
    stripped = content.strip()
    if not stripped:
        return []
    matches = HERMES_CALL.findall(stripped)
    if matches:
        return [_mapping(item, name="Hermes tool call") for item in matches]
    fenced = FENCED_JSON.findall(stripped)
    if fenced:
        return [_mapping(item, name="fenced action") for item in fenced]
    return [_mapping(stripped, name="model action")]


def parse_model_action(
    *, content: str | None, tool_calls: Sequence[Mapping[str, Any]] = ()
) -> AgentAction:
    """Parse exactly one action; parallel calls are unsafe for the stateful environment."""

    if tool_calls:
        if len(tool_calls) != 1:
            raise ValueError("exactly one sequential tool call is required per agent turn")
        return _parse_tool_call(tool_calls[0])

    objects = _content_objects(content or "")
    if len(objects) != 1:
        raise ValueError("the model must emit exactly one action per turn")
    payload = objects[0]
    if "name" in payload and "kind" not in payload:
        return _parse_tool_call(payload)
    return AgentAction.from_mapping(payload)

