"""Stable evidence records for documents, SQL, and analysis artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class EvidenceRecord:
    """A replayable pointer to one agent-visible observation."""

    evidence_id: str
    source_type: str
    source_ref: str
    source_version: str
    normalized_arguments: dict[str, Any]
    result_sha256: str
    data_snapshot_id: str
    tool_version: str

    @classmethod
    def build(
        cls,
        *,
        source_type: str,
        source_ref: str,
        source_version: str,
        arguments: dict[str, Any],
        result: Any,
        data_snapshot_id: str,
        tool_version: str,
    ) -> EvidenceRecord:
        """Create a deterministic ID from provenance, arguments, and result."""

        normalized_arguments = json.loads(_canonical_json(arguments))
        result_sha256 = hashlib.sha256(_canonical_json(result).encode("utf-8")).hexdigest()
        identity = {
            "source_type": source_type,
            "source_ref": source_ref,
            "source_version": source_version,
            "arguments": normalized_arguments,
            "result_sha256": result_sha256,
            "data_snapshot_id": data_snapshot_id,
            "tool_version": tool_version,
        }
        digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
        return cls(
            evidence_id=f"ev_{digest[:24]}",
            source_type=source_type,
            source_ref=source_ref,
            source_version=source_version,
            normalized_arguments=normalized_arguments,
            result_sha256=result_sha256,
            data_snapshot_id=data_snapshot_id,
            tool_version=tool_version,
        )
