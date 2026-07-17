"""Shared enumerations used by prompts and evaluation."""

from enum import Enum


class Subtrack(str, Enum):
    PERCEPTION_TEXT = "perception_text"
    PERCEPTION_GRAPHIC = "perception_graphic"
    FORMULATION_TEXT = "formulation_text"
    FORMULATION_GRAPHIC = "formulation_graphic"
    DEDUCTION = "deduction"


class InferenceProtocol(str, Enum):
    VIDEO = "video"
    IMAGE = "image"


ALL_SUBTRACKS = tuple(Subtrack)


def parse_subtrack(value: str) -> Subtrack:
    try:
        return Subtrack(value)
    except ValueError as exc:
        valid = ", ".join(subtrack.value for subtrack in Subtrack)
        raise ValueError(f"Unknown subtrack {value!r}. Expected one of: {valid}") from exc
