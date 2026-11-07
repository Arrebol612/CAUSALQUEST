"""Canonical actions, observations, and hashes for the CausalQuest agent loop."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


def canonical_json(payload: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_hash(payload: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class ActionKind(str, Enum):
    TOOL_CALL = "tool_call"
    STATE_UPDATE = "state_update"
    FINAL_REPORT = "final_report"


@dataclass(frozen=True)
class AgentAction:
    """One strictly validated action emitted by any policy or model adapter."""

    kind: ActionKind
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    patch: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    client_action_id: str | None = None
    reason: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AgentAction:
        if not isinstance(payload, Mapping):
            raise TypeError("action must be a mapping")
        try:
            kind = ActionKind(payload.get("kind"))
        except ValueError as exc:
            raise ValueError(f"unsupported action kind: {payload.get('kind')}") from exc

        action_id = payload.get("client_action_id")
        if action_id is not None and (not isinstance(action_id, str) or not action_id.strip()):
            raise ValueError("client_action_id must be a non-empty string when supplied")
        reason = payload.get("reason")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("reason must be a non-empty string when supplied")

        if kind is ActionKind.TOOL_CALL:
            tool_name = payload.get("tool_name")
            arguments = payload.get("arguments", {})
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("tool_call requires a non-empty tool_name")
            if not isinstance(arguments, Mapping):
                raise TypeError("tool_call arguments must be a mapping")
            return cls(
                kind=kind,
                tool_name=tool_name.strip(),
                arguments=dict(arguments),
                client_action_id=action_id,
                reason=reason,
            )

        if kind is ActionKind.STATE_UPDATE:
            patch = payload.get("patch")
            if not isinstance(patch, Mapping):
                raise TypeError("state_update requires a patch mapping")
            required = {"hypothesis_id", "status", "summary", "evidence_ids"}
            missing = sorted(required.difference(patch))
            unknown = sorted(set(patch).difference(required))
            if missing:
                raise ValueError(f"state_update patch is missing fields: {missing}")
            if unknown:
                raise ValueError(f"state_update patch has unsupported fields: {unknown}")
            if not isinstance(patch.get("evidence_ids"), list):
                raise TypeError("state_update evidence_ids must be a list")
            return cls(
                kind=kind,
                patch=dict(patch),
                client_action_id=action_id,
                reason=reason,
            )

        report = payload.get("report")
        if not isinstance(report, Mapping):
            raise TypeError("final_report requires a report mapping")
        return cls(
            kind=kind,
            report=dict(report),
            client_action_id=action_id,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind.value}
        if self.kind is ActionKind.TOOL_CALL:
            payload.update({"tool_name": self.tool_name, "arguments": self.arguments or {}})
        elif self.kind is ActionKind.STATE_UPDATE:
            payload["patch"] = self.patch or {}
        else:
            payload["report"] = self.report or {}
        if self.client_action_id is not None:
            payload["client_action_id"] = self.client_action_id
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class CaseSpec:
    """Evaluator-side case locator; its path is never included in policy observations."""

    case_id: str
    split: str
    task_type: str
    case_dir: str
    case_snapshot_id: str
    suite_snapshot_id: str
    oracle_sha256: str | None = None

    def public_identity(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "split": self.split,
            "task_type": self.task_type,
            "case_snapshot_id": self.case_snapshot_id,
            "suite_snapshot_id": self.suite_snapshot_id,
        }
