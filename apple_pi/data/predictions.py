"""Prediction path conventions for the required three rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apple_pi.constants import NUM_ROLLOUTS
from apple_pi.types import InferenceProtocol, Subtrack


@dataclass(frozen=True)
class PredictionRef:
    case_id: str
    protocol: InferenceProtocol
    subtrack: Subtrack
    rollout: int
    path: Path


def prediction_path(
    root: str | Path,
    case_id: str,
    protocol: InferenceProtocol,
    subtrack: Subtrack,
    rollout: int,
) -> Path:
    if rollout not in range(NUM_ROLLOUTS):
        raise ValueError(f"rollout must be 0..{NUM_ROLLOUTS - 1}")
    base = Path(root) / "cases" / Path(case_id) / subtrack.value
    if protocol == InferenceProtocol.VIDEO:
        return base / f"rollout_{rollout:02d}.mp4"
    if subtrack == Subtrack.DEDUCTION:
        return base / f"rollout_{rollout:02d}"
    return base / f"rollout_{rollout:02d}.png"
