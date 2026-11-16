"""Deterministic group-level dataset splitting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.70
    dev: float = 0.15
    public_test: float = 0.10
    locked_private_test: float = 0.05

    def __post_init__(self) -> None:
        values = (self.train, self.dev, self.public_test, self.locked_private_test)
        if any(value < 0 for value in values):
            raise ValueError("split fractions must be non-negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1.0")


def assign_group(
    group_key: str,
    *,
    namespace: str = "causalquest-v1",
    fractions: SplitFractions | None = None,
) -> str:
    """Assign a whole case family to one stable split."""

    if not group_key.strip():
        raise ValueError("group_key must not be empty")
    proportions = fractions or SplitFractions()
    payload = f"{namespace}\x1f{group_key}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64

    train_end = proportions.train
    dev_end = train_end + proportions.dev
    public_end = dev_end + proportions.public_test
    if bucket < train_end:
        return "train"
    if bucket < dev_end:
        return "dev"
    if bucket < public_end:
        return "public_test"
    return "locked_private_test"
