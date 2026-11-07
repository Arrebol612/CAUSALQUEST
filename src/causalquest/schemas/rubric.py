"""Typed rubric tree used by deterministic evaluators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RubricNode:
    """One weighted rubric node; child weights are local to their parent."""

    node_id: str
    weight: float
    description: str
    hard_gate: bool = False
    children: tuple[RubricNode, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("rubric node_id must not be empty")
        if not self.description.strip():
            raise ValueError(f"rubric node {self.node_id} needs a description")
        if self.weight < 0:
            raise ValueError(f"rubric node {self.node_id} has a negative weight")
        if self.children and abs(sum(child.weight for child in self.children) - 1.0) > 1e-9:
            raise ValueError(f"child weights for rubric node {self.node_id} must sum to 1")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RubricNode:
        return cls(
            node_id=str(payload.get("id", "")).strip(),
            weight=float(payload.get("weight", 0.0)),
            description=str(payload.get("description", "")).strip(),
            hard_gate=payload.get("hard_gate", False) is True,
            children=tuple(cls.from_mapping(item) for item in payload.get("children", [])),
        )

    def walk(self) -> tuple[RubricNode, ...]:
        return (self,) + tuple(descendant for child in self.children for descendant in child.walk())


@dataclass(frozen=True)
class RubricTree:
    """A versioned rubric with globally unique node identifiers."""

    version: str
    root: RubricNode

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("rubric version must not be empty")
        nodes = self.root.walk()
        ids = [node.node_id for node in nodes]
        duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate rubric node IDs: {duplicates}")
        if abs(self.root.weight - 1.0) > 1e-9:
            raise ValueError("root rubric weight must equal 1")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RubricTree:
        return cls(
            version=str(payload.get("version", "")).strip(),
            root=RubricNode.from_mapping(payload.get("root", {})),
        )
