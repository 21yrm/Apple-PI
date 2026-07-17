"""Build the exact prompt templates used by the paper implementation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from apple_pi.constants import NUM_ROLLOUTS, PROMPT_VERSION
from apple_pi.types import InferenceProtocol, Subtrack

from .templates import UM_PROMPT_TEMPLATES, VM_PROMPT_TEMPLATES

LETTERS = ("A", "B", "C", "D")


def _formula_choices_text(choices: Iterable[str], *, indent: bool) -> str:
    values = list(choices)
    if len(values) != 4:
        raise ValueError(
            f"formulation_text requires exactly four formula choices; got {len(values)}"
        )
    prefix = "  " if indent else ""
    return "\n".join(
        f"{prefix}{LETTERS[index]}) {formula}" for index, formula in enumerate(values)
    )


def build_prompt(
    protocol: InferenceProtocol | str,
    subtrack: Subtrack | str,
    *,
    formula_choices: Iterable[str] | None = None,
    target_time: float | None = None,
    physics_duration: float | None = None,
    time_point: float | None = None,
) -> str:
    """Render one paper prompt.

    The video protocol uses ``VM_PROMPT_TEMPLATES``. The image protocol uses
    ``UM_PROMPT_TEMPLATES``. No prompt wording is changed by this wrapper.
    """
    protocol = InferenceProtocol(protocol)
    subtrack = Subtrack(subtrack)
    templates = (
        VM_PROMPT_TEMPLATES
        if protocol == InferenceProtocol.VIDEO
        else UM_PROMPT_TEMPLATES
    )
    template = templates[subtrack]

    if subtrack == Subtrack.FORMULATION_TEXT:
        if formula_choices is None:
            raise ValueError("formula_choices is required for formulation_text")
        return template.format(
            formula_choices=_formula_choices_text(
                formula_choices,
                indent=protocol == InferenceProtocol.VIDEO,
            )
        )
    if subtrack == Subtrack.FORMULATION_GRAPHIC:
        if target_time is None:
            raise ValueError("target_time is required for formulation_graphic")
        return template.format(target_time=target_time)
    if subtrack == Subtrack.DEDUCTION:
        if protocol == InferenceProtocol.VIDEO:
            if physics_duration is None:
                raise ValueError("physics_duration is required for video Deduction")
            return template.format(physics_duration=int(physics_duration))
        if time_point is None:
            raise ValueError("time_point is required for image Deduction")
        return template.format(time_point=time_point)
    return template


def export_prompt_records(
    cases: Iterable[Mapping[str, Any]],
    protocol: InferenceProtocol | str,
) -> list[dict[str, Any]]:
    """Export a model-agnostic prompt manifest from public case metadata."""
    protocol = InferenceProtocol(protocol)
    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        for subtrack in Subtrack:
            common = {
                "case_id": case_id,
                "subtrack": subtrack.value,
                "protocol": protocol.value,
                "prompt_version": PROMPT_VERSION,
                "num_rollouts": NUM_ROLLOUTS,
                "input_image": case["input_image"],
            }
            if protocol == InferenceProtocol.IMAGE and subtrack == Subtrack.DEDUCTION:
                for time_point in case["deduction_timestamps"]:
                    records.append(
                        {
                            **common,
                            "time_point": time_point,
                            "prompt": build_prompt(
                                protocol,
                                subtrack,
                                time_point=time_point,
                            ),
                            "expected_output": "image",
                        }
                    )
                continue

            records.append(
                {
                    **common,
                    "prompt": build_prompt(
                        protocol,
                        subtrack,
                        formula_choices=case.get("formula_choices"),
                        target_time=case.get("target_time"),
                        physics_duration=case.get("physics_duration"),
                    ),
                    "expected_output": (
                        "video" if protocol == InferenceProtocol.VIDEO else "image"
                    ),
                }
            )
    return records
