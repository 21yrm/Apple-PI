"""LLM-based video evaluation against the 5 benchmark subtracks."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from apple_pi.types import Subtrack
from apple_pi.utils import bytes_to_data_url, frame_to_jpeg_bytes, parse_llm_json

logger = logging.getLogger(__name__)

class JudgeClient(Protocol):
    def chat_completion(self, messages: List[Dict[str, Any]]) -> str: ...


@dataclass
class EvaluationConfig:
    """Paper evaluation runtime configuration."""

    num_eval_frames: int = 8



@dataclass
class SubtrackScores:
    """Structured evaluation scores, each 0.0-1.0."""

    # Perception - Text (18 granular criteria)
    text_font_match: float = 0.0
    text_style_match: float = 0.0
    text_size_match: float = 0.0
    text_color_match: float = 0.0
    text_position_match: float = 0.0
    text_content_match: float = 0.0
    text_readability: float = 0.0
    axes_color_match: float = 0.0
    axes_style_match: float = 0.0
    axes_size_match: float = 0.0
    axes_position_match: float = 0.0
    axes_angle_match: float = 0.0
    velocity_color_match: float = 0.0
    velocity_style_match: float = 0.0
    velocity_size_match: float = 0.0
    velocity_position_match: float = 0.0
    no_extra_annotations: float = 0.0
    white_background: float = 0.0

    # Perception - Graphic
    coord_axes: float = 0.0
    velocity_arrows: float = 0.0
    text_annotations: float = 0.0
    objects_only_from_gt: float = 0.0
    object_completeness: float = 0.0
    object_visual_match: float = 0.0
    object_position_match: float = 0.0
    segmentation_iou: float = 0.0

    # Formulation - Text (formula)
    option_correct: float = 0.0
    formula_variables_correct: float = 0.0
    formula_constants_correct: float = 0.0
    formula_operators_correct: float = 0.0
    substitution_all_replaced: float = 0.0
    substitution_values_correct: float = 0.0
    substitution_unreplaced_correct: float = 0.0
    substitution_constants_correct: float = 0.0
    substitution_operators_correct: float = 0.0
    format_3_lines: float = 0.0
    background_pure_white: float = 0.0

    # Formulation - Graphic (target state)
    annotations_removed: float = 0.0
    object_position_match: float = 0.0
    object_appearance_match: float = 0.0
    arrow_is_borderless_orange: float = 0.0
    arrow_from_center: float = 0.0
    arrow_direction_match: float = 0.0
    velocity_label_present_and_format: float = 0.0
    velocity_value_match: float = 0.0
    velocity_label_no_unit: float = 0.0
    formulation_graphic_segmentation_iou: float = 0.0

    # Deduction - MLLM-based
    visual_quality: float = 0.0
    motion_smoothness: float = 0.0
    physics_accuracy: float = 0.0
    gen_annotations_removed: float = 0.0
    object_consistency: float = 0.0

    # Deduction - Programmatic Metrics (raw values; -1.0 = not computed)
    gen_psnr: float = -1.0
    gen_masked_psnr: float = -1.0
    gen_spatial_iou: float = -1.0
    gen_spatiotemporal_iou: float = -1.0
    gen_weighted_spatial_iou: float = -1.0
    gen_velocity_error: float = -1.0

    feedback: str = ""

    def _mean(self, vals: list) -> float:
        valid_vals = [v for v in vals if v >= 0.0]
        return float(np.mean(valid_vals)) if valid_vals else 0.0

    def _psnr_score(self, psnr: float, max_db: float = 40.0) -> float:
        """Normalize PSNR (dB) to [0, 1]. Returns -1.0 if not computed."""
        if psnr < 0.0:
            return -1.0
        return float(np.clip(psnr / max_db, 0.0, 1.0))

    def _velocity_accuracy(self, error: float) -> float:
        """Convert velocity error to an accuracy score in [0, 1] via 1/(1+error).
        Returns -1.0 if not computed."""
        if error < 0.0:
            return -1.0
        return float(1.0 / (1.0 + error))

    def perception_text_score(self) -> float:
        content_score = self._mean([
            self.text_content_match,
            self.text_readability,
            self.no_extra_annotations,
        ])

        layout_score = self._mean([
            self.text_position_match,
            self.axes_position_match,
            self.axes_angle_match,
            self.velocity_position_match,
        ])

        style_score = self._mean([
            self.text_font_match,
            self.text_style_match,
            self.text_size_match,
            self.text_color_match,
            self.axes_color_match,
            self.axes_style_match,
            self.axes_size_match,
            self.velocity_color_match,
            self.velocity_style_match,
            self.velocity_size_match,
            self.white_background,
        ])

        return 0.5 * content_score + 0.3 * layout_score + 0.2 * style_score

    def perception_graphic_score(self) -> float:
        background_score = self._mean([self.white_background])
        annotation_removal_score = self._mean([
            self.coord_axes,
            self.velocity_arrows,
            self.text_annotations,
        ])
        object_preservation_score = self._mean([
            self.objects_only_from_gt,
            self.object_completeness,
            self.object_visual_match,
        ])
        spatial_score = self._mean([
            self.object_position_match,
            self.segmentation_iou,
        ])

        groups = {
            "background": (background_score, 0.10),
            "annotation_removal": (annotation_removal_score, 0.30),
            "object_preservation": (object_preservation_score, 0.30),
            "spatial_consistency": (spatial_score, 0.30),
        }

        valid = {k: (s, w) for k, (s, w) in groups.items() if s >= 0.0 and w > 0.0}
        denom = sum(w for _, w in valid.values())
        if denom <= 0.0:
            return 0.0
        return float(sum(s * (w / denom) for s, w in valid.values()))

    def formulation_text_score(self) -> float:
        # Grouped reporting + weighted average.
        option_score = self._mean([self.option_correct])
        formula_score = self._mean([
            self.formula_variables_correct,
            self.formula_constants_correct,
            self.formula_operators_correct,
        ])
        substitution_score = self._mean([
            self.substitution_all_replaced,
            self.substitution_values_correct,
            self.substitution_unreplaced_correct,
            self.substitution_constants_correct,
            self.substitution_operators_correct,
        ])
        presentation_score = self._mean([
            self.format_3_lines,
            self.background_pure_white,
        ])

        groups = {
            "option": (option_score, 0.20),
            "formula": (formula_score, 0.30),
            "substitution": (substitution_score, 0.40),
            "presentation": (presentation_score, 0.10),
        }

        valid = {
            k: (s, w) for k, (s, w) in groups.items() if s >= 0.0 and w > 0.0
        }
        denom = sum(w for _, w in valid.values())
        if denom <= 0.0:
            return 0.0
        return float(sum(s * (w / denom) for s, w in valid.values()))

    def formulation_graphic_score(self) -> float:
        # Grouped reporting + weighted average.
        annotation_removal_score = self._mean([self.annotations_removed])
        object_match_score = self._mean(
            [self.object_position_match, self.object_appearance_match,
             self.formulation_graphic_segmentation_iou]
        )
        arrow_quality_score = self._mean(
            [
                self.arrow_is_borderless_orange,
                self.arrow_from_center,
                self.arrow_direction_match,
            ]
        )
        velocity_text_score = self._mean(
            [
                self.velocity_label_present_and_format,
                self.velocity_value_match,
                self.velocity_label_no_unit,
            ]
        )

        groups = {
            "annotation_removal": (annotation_removal_score, 0.1),
            "object_match": (object_match_score, 0.30),
            "arrow_quality": (arrow_quality_score, 0.30),
            "velocity_text": (velocity_text_score, 0.30),
        }

        valid = {
            k: (s, w) for k, (s, w) in groups.items() if s >= 0.0 and w > 0.0
        }
        denom = sum(w for _, w in valid.values())
        if denom <= 0.0:
            return 0.0
        return float(sum(s * (w / denom) for s, w in valid.values()))

    def deduction_score(self) -> float:
        return self.deduction_weighted_score()

    def deduction_category_scores(self) -> Dict[str, float]:
        """
        The Deduction subtrack is judged by three aspects:

        1) Integrity: annotations removed + no object hallucination/disappearance.
           - MLLM: gen_annotations_removed, object_consistency

        2) Fidelity: visual quality + temporal smoothness + pixel-level accuracy.
           - MLLM: visual_quality, motion_smoothness
           - Prog: PSNR score (gen_psnr normalized to [0,1]),
                   MaskedPSNR score (gen_masked_psnr normalized to [0,1])

        3) Physics: physical motion correctness + spatial trajectory accuracy.
           - MLLM: physics_accuracy
           - Prog: gen_spatial_iou, gen_spatiotemporal_iou,
                   gen_weighted_spatial_iou,
                   velocity accuracy (1 / (1 + gen_velocity_error))

        Programmatic metrics default to -1.0 when not computed and are
        excluded from the mean automatically.
        """
        annotation_entity_integrity_score = self._mean(
            [self.gen_annotations_removed, self.object_consistency]
        )
        visual_motion_fidelity_score = self._mean(
            [
                self.visual_quality,
                self.motion_smoothness,
                self._psnr_score(self.gen_psnr),
                self._psnr_score(self.gen_masked_psnr),
            ]
        )
        physics_correctness_score = self._mean(
            [
                self.physics_accuracy,
                self.gen_spatial_iou,
                self.gen_spatiotemporal_iou,
                self.gen_weighted_spatial_iou,
                self._velocity_accuracy(self.gen_velocity_error),
            ]
        )

        return {
            "annotation_entity_integrity": annotation_entity_integrity_score,
            "visual_motion_fidelity": visual_motion_fidelity_score,
            "physics_correctness": physics_correctness_score,
        }

    def deduction_weighted_score(self) -> float:
        # Weights reflect importance for Deduction:
        # - Physics correctness is the primary goal.
        # - Annotation/object integrity is required for correctness of the rendered scene.
        # - Visual/temporal fidelity is important but should not override physical correctness.
        category_weights = {
            "annotation_entity_integrity": 0.20,
            "visual_motion_fidelity": 0.20,
            "physics_correctness": 0.60,
        }
        cats = self.deduction_category_scores()
        return float(
            sum(cats[k] * w for k, w in category_weights.items())
        )

    def subtrack_score(self, subtrack: Subtrack) -> float:
        return {
            Subtrack.PERCEPTION_TEXT: self.perception_text_score,
            Subtrack.PERCEPTION_GRAPHIC: self.perception_graphic_score,
            Subtrack.FORMULATION_TEXT: self.formulation_text_score,
            Subtrack.FORMULATION_GRAPHIC: self.formulation_graphic_score,
            Subtrack.DEDUCTION: self.deduction_score,
        }[subtrack]()

    def to_dict(self) -> dict:
        content_score = self._mean([
            self.text_content_match,
            self.text_readability,
            self.no_extra_annotations,
        ])

        layout_score = self._mean([
            self.text_position_match,
            self.axes_position_match,
            self.axes_angle_match,
            self.velocity_position_match,
        ])

        style_score = self._mean([
            self.text_font_match,
            self.text_style_match,
            self.text_size_match,
            self.text_color_match,
            self.axes_color_match,
            self.axes_style_match,
            self.axes_size_match,
            self.velocity_color_match,
            self.velocity_style_match,
            self.velocity_size_match,
            self.white_background,
        ])

        return {
            "perception_text": {
                "text_font_match": self.text_font_match,
                "text_style_match": self.text_style_match,
                "text_size_match": self.text_size_match,
                "text_color_match": self.text_color_match,
                "text_position_match": self.text_position_match,
                "text_content_match": self.text_content_match,
                "text_readability": self.text_readability,
                "axes_color_match": self.axes_color_match,
                "axes_style_match": self.axes_style_match,
                "axes_size_match": self.axes_size_match,
                "axes_position_match": self.axes_position_match,
                "axes_angle_match": self.axes_angle_match,
                "velocity_color_match": self.velocity_color_match if self.velocity_color_match >= 0.0 else None,
                "velocity_style_match": self.velocity_style_match if self.velocity_style_match >= 0.0 else None,
                "velocity_size_match": self.velocity_size_match if self.velocity_size_match >= 0.0 else None,
                "velocity_position_match": self.velocity_position_match if self.velocity_position_match >= 0.0 else None,
                "no_extra_annotations": self.no_extra_annotations,
                "white_background": self.white_background,
                "content_score": round(content_score, 3),
                "layout_score": round(layout_score, 3),
                "style_score": round(style_score, 3),
                "avg": round(self.perception_text_score(), 3),
            },
            "perception_graphic": {
                "white_background": self.white_background,
                "coord_axes": self.coord_axes,
                "velocity_arrows": self.velocity_arrows if self.velocity_arrows >= 0.0 else None,
                "text_annotations": self.text_annotations,
                "objects_only_from_gt": self.objects_only_from_gt,
                "object_completeness": self.object_completeness,
                "object_visual_match": self.object_visual_match,
                "object_position_match": self.object_position_match,
                "segmentation_iou": round(self.segmentation_iou, 3),
                "background_score": round(self._mean([self.white_background]), 3),
                "annotation_removal_score": round(self._mean([
                    self.coord_axes,
                    self.velocity_arrows,
                    self.text_annotations,
                ]), 3),
                "object_preservation_score": round(self._mean([
                    self.objects_only_from_gt,
                    self.object_completeness,
                    self.object_visual_match,
                ]), 3),
                "spatial_consistency_score": round(self._mean([
                    self.object_position_match,
                    self.segmentation_iou,
                ]), 3),
                "avg": round(self.perception_graphic_score(), 3),
            },
            "formulation_text": {
                "option_correct": self.option_correct,
                "formula_variables_correct": self.formula_variables_correct,
                "formula_constants_correct": self.formula_constants_correct,
                "formula_operators_correct": self.formula_operators_correct,
                "substitution_all_replaced": self.substitution_all_replaced,
                "substitution_values_correct": self.substitution_values_correct,
                "substitution_unreplaced_correct": self.substitution_unreplaced_correct,
                "substitution_constants_correct": self.substitution_constants_correct,
                "substitution_operators_correct": self.substitution_operators_correct,
                "format_3_lines": self.format_3_lines,
                "background_pure_white": self.background_pure_white,
                "option_score": round(self._mean([self.option_correct]), 3),
                "formula_score": round(self._mean([
                    self.formula_variables_correct,
                    self.formula_constants_correct,
                    self.formula_operators_correct,
                ]), 3),
                "substitution_score": round(self._mean([
                    self.substitution_all_replaced,
                    self.substitution_values_correct,
                    self.substitution_unreplaced_correct,
                    self.substitution_constants_correct,
                    self.substitution_operators_correct,
                ]), 3),
                "presentation_score": round(self._mean([
                    self.format_3_lines,
                    self.background_pure_white,
                ]), 3),
                "avg": round(self.formulation_text_score(), 3),
            },
            "formulation_graphic": {
                "annotations_removed": self.annotations_removed,
                "object_position_match": self.object_position_match,
                "object_appearance_match": self.object_appearance_match,
                "formulation_graphic_segmentation_iou": round(self.formulation_graphic_segmentation_iou, 3),
                "arrow_is_borderless_orange": self.arrow_is_borderless_orange,
                "arrow_from_center": self.arrow_from_center,
                "arrow_direction_match": self.arrow_direction_match,
                "velocity_label_present_and_format": self.velocity_label_present_and_format,
                "velocity_value_match": self.velocity_value_match,
                "velocity_label_no_unit": self.velocity_label_no_unit,
                "annotation_removal_score": round(
                    self._mean([self.annotations_removed]), 3
                ),
                "object_match_score": round(
                    self._mean(
                        [self.object_position_match, self.object_appearance_match,
                         self.formulation_graphic_segmentation_iou]
                    ),
                    3,
                ),
                "arrow_quality_score": round(
                    self._mean(
                        [
                            self.arrow_is_borderless_orange,
                            self.arrow_from_center,
                            self.arrow_direction_match,
                        ]
                    ),
                    3,
                ),
                "velocity_text_score": round(
                    self._mean(
                        [
                            self.velocity_label_present_and_format,
                            self.velocity_value_match,
                            self.velocity_label_no_unit,
                        ]
                    ),
                    3,
                ),
                "avg": round(self.formulation_graphic_score(), 3),
            },
            "deduction": {
                # ── MLLM-based dimensions ──────────────────────────────────
                # integrity
                "gen_annotations_removed": self.gen_annotations_removed,
                "object_consistency": self.object_consistency,
                # fidelity
                "visual_quality": self.visual_quality,
                "motion_smoothness": self.motion_smoothness,
                # physics
                "physics_accuracy": self.physics_accuracy,
                # ── Programmatic metrics (raw values) ─────────────────────
                # fidelity
                "psnr": round(self.gen_psnr, 3) if self.gen_psnr >= 0 else None,
                "masked_psnr": round(self.gen_masked_psnr, 3) if self.gen_masked_psnr >= 0 else None,
                "psnr_score": round(self._psnr_score(self.gen_psnr), 3) if self.gen_psnr >= 0 else None,
                "masked_psnr_score": round(self._psnr_score(self.gen_masked_psnr), 3) if self.gen_masked_psnr >= 0 else None,
                # physics
                "spatial_iou": round(self.gen_spatial_iou, 3) if self.gen_spatial_iou >= 0 else None,
                "spatiotemporal_iou": round(self.gen_spatiotemporal_iou, 3) if self.gen_spatiotemporal_iou >= 0 else None,
                "weighted_spatial_iou": round(self.gen_weighted_spatial_iou, 3) if self.gen_weighted_spatial_iou >= 0 else None,
                "velocity_error": round(self.gen_velocity_error, 3) if self.gen_velocity_error >= 0 else None,
                "velocity_accuracy": round(self._velocity_accuracy(self.gen_velocity_error), 3) if self.gen_velocity_error >= 0 else None,
                # ── Category scores (MLLM + programmatic fused) ───────────
                "annotation_entity_integrity_score": round(
                    self.deduction_category_scores()["annotation_entity_integrity"],
                    3,
                ),
                "visual_motion_fidelity_score": round(
                    self.deduction_category_scores()["visual_motion_fidelity"],
                    3,
                ),
                "physics_correctness_score": round(
                    self.deduction_category_scores()["physics_correctness"],
                    3,
                ),
                "avg": round(self.deduction_score(), 3),
            },
            "feedback": self.feedback,
        }


# ── Evaluation prompt templates ────────────────────────────────────────

EVAL_PROMPT_PERCEPTION_TEXT = """\
You are an expert evaluator assessing how perfectly a generated image preserves the exact annotation details from a ground truth reference image.

You are provided with two images:
1. "REFERENCE GROUND TRUTH IMAGE": The perfectly correct original image with annotations on a pure white background.
2. "GENERATED IMAGE": The image to be evaluated.

Your task is to compare the GENERATED IMAGE against the REFERENCE GROUND TRUTH IMAGE and score the generated image across 18 distinct criteria.
For each criterion, assign a score between 0.0 (completely unmatched/wrong) and 1.0 (perfectly matched).

Evaluate the following criteria carefully:

1. text_font_match: Does every text annotation use the EXACT SAME font family/typeface as the ground truth?
2. text_style_match: Does every text annotation have the EXACT SAME style (e.g., bold, italic, regular) as the ground truth?
3. text_size_match: Does every text annotation have the EXACT SAME font size/scale as the ground truth?
4. text_color_match: Does every text annotation have the EXACT SAME color as the ground truth?
5. text_position_match: Is every text annotation located in the EXACT SAME position as the ground truth?
6. text_content_match: Does the textual content of every single annotation EXACTLY match the ground truth (no typos, missing words, or hallucinations)?
7. text_readability: Are all text labels and annotations perfectly crisp, sharp, and easy to read, without any blurriness, distortion, or generation artifacts, exactly matching the clarity of the ground truth? (0.0=heavily blurred/distorted, 1.0=perfectly crisp and readable).
8. axes_color_match: Is the color of the coordinate axes arrows EXACTLY the same as the ground truth?
9. axes_style_match: Is the line style (solid vs dashed) of the coordinate axes arrows EXACTLY the same as the ground truth?
10. axes_size_match: Are the width, length, and arrowhead size of the coordinate axes arrows EXACTLY the same as the ground truth?
11. axes_position_match: Is the position of the coordinate axes EXACTLY the same as the ground truth?
12. axes_angle_match: Are the angles between the coordinate axes EXACTLY the same as the ground truth?
13. velocity_color_match: If initial velocity arrows exist in the GT, does the generated image have velocity arrows with the EXACT SAME color? (If no velocity arrows exist in GT, score -1.0)
14. velocity_style_match: If initial velocity arrows exist in the GT, do they have the EXACT SAME line style (solid vs dashed)? (If no velocity arrows exist in GT, score -1.0)
15. velocity_size_match: If initial velocity arrows exist in the GT, do they have the EXACT SAME width, length, and arrowhead size? (If no velocity arrows exist in GT, score -1.0)
16. velocity_position_match: If initial velocity arrows exist in the GT, are they in the EXACT SAME starting position? (If no velocity arrows exist in GT, score -1.0)
17. no_extra_annotations: Does the generated image only contain the annotations present in the GT, with absolutely NO extra, hallucinated, or duplicated annotations? (0.0=extra stuff exists, 1.0=clean, matches GT exactly).
18. white_background: Is the background of the entire generated image pure white, just like the ground truth?

Respond with ONLY JSON:
{{
  "text_font_match": <float>,
  "text_style_match": <float>,
  "text_size_match": <float>,
  "text_color_match": <float>,
  "text_position_match": <float>,
  "text_content_match": <float>,
  "text_readability": <float>,
  "axes_color_match": <float>,
  "axes_style_match": <float>,
  "axes_size_match": <float>,
  "axes_position_match": <float>,
  "axes_angle_match": <float>,
  "velocity_color_match": <float>,
  "velocity_style_match": <float>,
  "velocity_size_match": <float>,
  "velocity_position_match": <float>,
  "no_extra_annotations": <float>,
  "white_background": <float>,
  "feedback": "<2-4 sentences summarizing any deviations found across these criteria>"
}}"""

EVAL_PROMPT_PERCEPTION_GRAPHIC = """\
You are an expert image evaluator.

You are provided with THREE images:
1. "REFERENCE WHITE BACKGROUND ANNOTATION IMAGE": a ground-truth reference on a pure white background that contains human annotations (coordinate axes, text labels, and possibly initial-velocity arrows).
2. "REFERENCE WHITE BACKGROUND OBJECT IMAGE": a ground-truth reference on a pure white background that contains ONLY the physical objects (no annotations, no background/ground/environment).
3. "GENERATED IMAGE": the image to be evaluated.

Your job is to score how well the GENERATED IMAGE matches the intended "objects-only on pure white" result, using the two reference images to understand what should be removed and what should remain.

Score each metric from 0.0 to 1.0 (higher is better). Be strict: partial visibility, faint remnants, blur, or mismatched appearance should reduce the score.

BACKGROUND (purity):
- white_background: Is the entire background in the GENERATED IMAGE truly pure white (no gradients, shadows, textures, ground plane, or environmental remnants)?

ANNOTATION REMOVAL (compare against the annotation reference):
- coord_axes: Are coordinate axes and axis labels fully removed? (0=clearly visible, 1=completely gone)
- velocity_arrows: Are any initial-velocity arrows fully removed? (0=visible, 1=gone. If NO velocity arrows exist in the annotation reference, return -1.0.)
- text_annotations: Are ALL text labels, numbers, and other textual annotations removed? (0=visible, 1=gone)

OBJECT PRESERVATION (compare against the object reference):
- objects_only_from_gt: Does the GENERATED IMAGE contain ONLY the physical objects present in the object reference (no extra objects, no missing objects, and no leftover ground/background surfaces)?
- object_completeness: Are all objects from the object reference present and uncut/undamaged in the GENERATED IMAGE (no missing parts, no erasures, no merging into the background)?
- object_visual_match: Do the kept objects match the object reference visually (shape/geometry, relative size, colors, textures/materials, shading/lighting cues, and any distinctive visual details)? (0=looks different, 1=visually identical)
- object_position_match: Do the objects appear at the EXACT SAME image positions as in the object reference (same 2D placement and relative arrangement; no shifting, drifting, or rearrangement)?

Respond with ONLY JSON:
{{
  "white_background": <float>,
  "coord_axes": <float>,
  "velocity_arrows": <float>,
  "text_annotations": <float>,
  "objects_only_from_gt": <float>,
  "object_completeness": <float>,
  "object_visual_match": <float>,
  "object_position_match": <float>,
  "feedback": "<2-4 sentences summarizing the biggest issues>"
}}"""

EVAL_PROMPT_FORMULATION_TEXT = """\
You are evaluating a model's physical comprehension.

The model was asked to output an image containing 3 text lines on a pure white background:
Line 1: The correct option letter.
Line 2: The correct formula.
Line 3: The formula with annotated variables replaced by corresponding values. If the scene has multiple objects, values belonging to specific objects must come from exactly ONE consistent set (e.g., one object for single-body motion, or multiple objects forming one valid set for multiple-body motion).

Correct Formula is {correct_letter}: {correct_formula}
Annotation Content: {annotation}

Evaluate the generated image (Score 0.0-1.0):

- option_correct: Is the correct letter (Line 1) chosen?
- formula_variables_correct: Are physical symbols in the formula (Line 2) written correctly?
- formula_constants_correct: Are inherent numeric constants in the formula written correctly?
- formula_operators_correct: Are operators in the formula (Line 2) written correctly?
- substitution_all_replaced: Are ALL annotated variables replaced by values in Line 3?
- substitution_values_correct: Are substituted values correct? If there are multiple objects, did the model pick exactly one valid set of objects to substitute values?
- substitution_unreplaced_correct: Are unannotated symbols correctly kept unchanged in Line 3?
- substitution_constants_correct: Are inherent numeric constants correctly kept in Line 3?
- substitution_operators_correct: Are operators correctly written in Line 3?
- format_3_lines: Does it strictly follow the 3-line format?
- background_pure_white: Is the background pure white?

Respond with ONLY JSON:
{{
  "option_correct": <float>,
  "formula_variables_correct": <float>,
  "formula_constants_correct": <float>,
  "formula_operators_correct": <float>,
  "substitution_all_replaced": <float>,
  "substitution_values_correct": <float>,
  "substitution_unreplaced_correct": <float>,
  "substitution_constants_correct": <float>,
  "substitution_operators_correct": <float>,
  "format_3_lines": <float>,
  "background_pure_white": <float>,
  "feedback": "<2-3 sentences summarizing issues>"
}}"""

EVAL_PROMPT_FORMULATION_GRAPHIC = """\
You are evaluating a generated image against a ground truth (GT) reference image \
and an initial annotated scene image.

The images are provided in this order:
1. REFERENCE GROUND TRUTH IMAGE (showing the correct physical positions of all objects, \
their velocity arrow directions, and their instantaneous speed values).
2. INITIAL ANNOTATED PHYSICS SCENE (showing objects, axes, and all initial text labels).
3. GENERATED IMAGE.

Please evaluate the GENERATED IMAGE based on the following metrics. Score each from 0.0 to 1.0. \
If there are multiple objects, your score should reflect the average performance across ALL objects:

- annotations_removed: Are all annotations from the INITIAL ANNOTATED PHYSICS SCENE \
(coordinate axes, text labels, initial velocity arrows) correctly removed in the \
GENERATED IMAGE? (0=not removed at all, 1=completely clean)
- object_position_match: Are the positions of all objects in the GENERATED IMAGE exactly \
the same as their corresponding positions in the GT reference image? \
(0=completely different, 1=exact match for all objects)
- object_appearance_match: Do all objects in the GENERATED IMAGE look visually \
identical to the corresponding objects in the GT reference image, with special \
attention to EXACT texture/material detail and EXACT object size/scale? \
(0=very different, 1=identical for all objects)
- arrow_is_borderless_orange: Are the instantaneous velocity arrows for all moving objects \
in the GENERATED IMAGE borderless orange arrows? (0=no/wrong style/color for all, 1=yes for all)
- arrow_from_center: Do the instantaneous velocity arrows in the GENERATED IMAGE \
originate from the centers of their respective objects? (0=no for all, 1=yes for all)
- arrow_direction_match: Are the directions of the instantaneous velocity arrows in the \
GENERATED IMAGE completely consistent with the corresponding velocity arrow directions \
in the GT reference image? (0=completely wrong directions, 1=exact match for all moving objects)
- velocity_label_present_and_format: Does EVERY object in the GENERATED IMAGE have a \
velocity text label that roughly follows the pattern "v = number" (or "v = 0" if at rest)? \
The exact spacing/decimals do NOT need to match; the key is that a clear velocity label exists. \
(0=missing labels for all objects, 1=labels present for all objects)
- velocity_value_match: How closely do the numerical values of the velocity text labels in \
the GENERATED IMAGE match the values in the GT reference image? \
(0=values are completely wrong or missing, 1=values match exactly for all objects. \
Give partial credit, e.g., 0.1-0.9, depending on how numerically close the values are \
and how many objects are correct)
- velocity_label_no_unit: Do the velocity text labels in the GENERATED IMAGE \
contain only numbers WITHOUT any units (e.g., no "m/s" or similar)? \
(0=all have units, 1=no units for any object)

Respond with ONLY JSON:
{{
  "annotations_removed": <float>,
  "object_position_match": <float>,
  "object_appearance_match": <float>,
  "arrow_is_borderless_orange": <float>,
  "arrow_from_center": <float>,
  "arrow_direction_match": <float>,
  "velocity_label_present_and_format": <float>,
  "velocity_value_match": <float>,
  "velocity_label_no_unit": <float>,
  "feedback": "<2-4 sentences summarizing issues>"
}}"""

EVAL_PROMPT_DEDUCTION = """\
You are evaluating a physics video generation model's ability to generate \
physically accurate and visually high-quality motion.

You are provided with:
1. INITIAL ANNOTATED PHYSICS SCENE (the first frame, showing objects, coordinate axes, and annotated initial conditions).
2. GENERATED VIDEO (the model's output).
{gt_section}

Please evaluate the GENERATED VIDEO based on the following metrics. Score each from 0.0 to 1.0:

- gen_annotations_removed: Are all human annotations from the INITIAL SCENE \
(such as text labels, coordinate axes, and initial velocity arrows) successfully removed \
in the generated video? (0 = annotations remain visible or leave severe ghostly artifacts, \
1 = perfectly clean removal, leaving only the physical objects and background).
- object_consistency: Do the physical objects remain perfectly consistent throughout the video? \
(0 = objects suddenly disappear, new objects appear out of nowhere, or objects undergo severe unintended deformation, \
1 = objects maintain their exact identity, shape, and count perfectly).
- visual_quality: What is the overall visual quality of the generated video? \
Consider sharpness, clarity, and absence of visual artifacts or noise. \
(0 = severe artifacts or blur, 1 = high-quality, crisp rendering).
- motion_smoothness: Is the temporal motion of the objects smooth and natural? \
(0 = severely broken motion, flickering, stuttering, or teleporting, \
1 = perfectly smooth, continuous, and natural movement).
- physics_accuracy: Does the motion of the objects accurately follow the physics described \
by the initial conditions (trajectory, velocity, gravity, collisions, etc.)? \
(0 = completely defies physics, 1 = physically accurate motion).

Respond with ONLY JSON:
{{
  "gen_annotations_removed": <float>,
  "object_consistency": <float>,
  "visual_quality": <float>,
  "motion_smoothness": <float>,
  "physics_accuracy": <float>,
  "feedback": "<2-4 sentences summarizing issues regarding annotation removal, visual quality, object consistency, smoothness, and physics.>"
}}"""

GT_COMPARISON_SECTION = """
3. GROUND TRUTH FRAMES showing the correct physics are also provided. \
Compare generated vs GT: object positions, trajectories, timing."""

EVAL_PROMPTS = {
    Subtrack.PERCEPTION_TEXT: EVAL_PROMPT_PERCEPTION_TEXT,
    Subtrack.PERCEPTION_GRAPHIC: EVAL_PROMPT_PERCEPTION_GRAPHIC,
    Subtrack.FORMULATION_TEXT: EVAL_PROMPT_FORMULATION_TEXT,
    Subtrack.FORMULATION_GRAPHIC: EVAL_PROMPT_FORMULATION_GRAPHIC,
    Subtrack.DEDUCTION: EVAL_PROMPT_DEDUCTION,
}

SUBTRACK_FIELDS = {
    Subtrack.PERCEPTION_TEXT: [
        "text_font_match",
        "text_style_match",
        "text_size_match",
        "text_color_match",
        "text_position_match",
        "text_content_match",
        "text_readability",
        "axes_color_match",
        "axes_style_match",
        "axes_size_match",
        "axes_position_match",
        "axes_angle_match",
        "velocity_color_match",
        "velocity_style_match",
        "velocity_size_match",
        "velocity_position_match",
        "no_extra_annotations",
        "white_background",
    ],
    Subtrack.PERCEPTION_GRAPHIC: [
        "white_background",
        "coord_axes",
        "velocity_arrows",
        "text_annotations",
        "objects_only_from_gt",
        "object_completeness",
        "object_visual_match",
        "object_position_match",
    ],
    Subtrack.FORMULATION_TEXT: [
        "option_correct",
        "formula_variables_correct",
        "formula_constants_correct",
        "formula_operators_correct",
        "substitution_all_replaced",
        "substitution_values_correct",
        "substitution_unreplaced_correct",
        "substitution_constants_correct",
        "substitution_operators_correct",
        "format_3_lines",
        "background_pure_white",
    ],
    Subtrack.FORMULATION_GRAPHIC: [
        "annotations_removed",
        "object_position_match",
        "object_appearance_match",
        "arrow_is_borderless_orange",
        "arrow_from_center",
        "arrow_direction_match",
        "velocity_label_present_and_format",
        "velocity_value_match",
        "velocity_label_no_unit",
    ],
    Subtrack.DEDUCTION: [
        "visual_quality",
        "motion_smoothness",
        "physics_accuracy",
        "gen_annotations_removed",
        "object_consistency",
    ],
}


class VideoEvaluator:
    """Evaluates generated videos using LLM (Gemini 3.0 Flash)."""

    def __init__(self, client: JudgeClient, opt_config: EvaluationConfig):
        self.client = client
        self.opt_config = opt_config

    def evaluate(
        self,
        subtrack: Subtrack,
        first_frame_data_url: str,
        gen_frames: List[np.ndarray],
        gen_video_bytes: Optional[bytes] = None,
        gt_frames: Optional[List[np.ndarray]] = None,
        subtrack_ref_data_url: Optional[str] = None,
        formula_info: Optional[Dict[str, str]] = None,
        first_frame_white_bg_data_url: Optional[str] = None,
        first_frame_white_bg_obj_data_url: Optional[str] = None,
    ) -> SubtrackScores:
        """Evaluate generated video frames against a subtrack.

        Args:
            subtrack: Which subtrack to evaluate.
            first_frame_data_url: data URL of the annotated first frame.
            gen_frames: sampled frames from the generated video.
            gen_video_bytes: for Subtrack.DEDUCTION, raw mp4 bytes to pass
                directly to the model (instead of per-frame sampling).
            gt_frames: optional ground truth frames for comparison.
            subtrack_ref_data_url: reference image for moment subtrack.
            formula_info: dict with correct_letter, correct_formula, annotation
                for formulation_text evaluation.
        """
        eval_prompt = EVAL_PROMPTS[subtrack]

        # Fill in template variables
        if subtrack == Subtrack.DEDUCTION:
            gt_section = GT_COMPARISON_SECTION if gt_frames else ""
            eval_prompt = eval_prompt.format(gt_section=gt_section)
        elif subtrack == Subtrack.FORMULATION_TEXT and formula_info:
            eval_prompt = eval_prompt.format(**formula_info)

        # Build multimodal message
        if subtrack == Subtrack.PERCEPTION_TEXT:
            assert first_frame_white_bg_data_url
            content: List[Dict[str, Any]] = [
                {"type": "text", "text": eval_prompt},
                {"type": "text", "text": "REFERENCE GROUND TRUTH IMAGE:"},
                {"type": "image_url", "image_url": {"url": first_frame_white_bg_data_url}},
            ]
        elif subtrack == Subtrack.PERCEPTION_GRAPHIC:
            assert first_frame_white_bg_data_url
            assert first_frame_white_bg_obj_data_url
            content: List[Dict[str, Any]] = [
                {"type": "text", "text": eval_prompt},
                {"type": "text", "text": "REFERENCE WHITE BACKGROUND ANNOTATION IMAGE:"},
                {"type": "image_url", "image_url": {"url": first_frame_white_bg_data_url}},
                {"type": "text", "text": "REFERENCE WHITE BACKGROUND OBJECT IMAGE:"},
                {"type": "image_url", "image_url": {"url": first_frame_white_bg_obj_data_url}},
            ]
        elif subtrack in [Subtrack.FORMULATION_TEXT, Subtrack.DEDUCTION]:
            content: List[Dict[str, Any]] = [
                {"type": "text", "text": eval_prompt},
                {"type": "text", "text": "INITIAL ANNOTATED PHYSICS SCENE (showing objects, axes, and all text labels):"},
                {"type": "image_url", "image_url": {"url": first_frame_data_url}},
            ]
        elif subtrack == Subtrack.FORMULATION_GRAPHIC:
            content: List[Dict[str, Any]] = [
                {"type": "text", "text": eval_prompt},
                {"type": "text", "text": "REFERENCE GROUND TRUTH IMAGE (showing the object's correct physical position, velocity arrow direction, and instantaneous speed value):"},
                {"type": "image_url", "image_url": {"url": subtrack_ref_data_url}},
                {"type": "text", "text": "INITIAL ANNOTATED PHYSICS SCENE (showing objects, axes, and all text labels):"},
                {"type": "image_url", "image_url": {"url": first_frame_data_url}},
            ]

        # Generated frames
        if subtrack != Subtrack.DEDUCTION:
            content.append({
                "type": "text",
                "text": "GENERATED IMAGE:",
            })
            assert len(gen_frames) == 1
            jpeg = frame_to_jpeg_bytes(gen_frames[0])
            content.append({
                "type": "image_url",
                "image_url": {"url": bytes_to_data_url(jpeg)},
            })
        else:
            assert gen_video_bytes is not None, (
                "Subtrack.DEDUCTION requires gen_video_bytes (mp4 bytes)."
            )
            content.append({
                "type": "text",
                "text": "GENERATED VIDEO:",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": bytes_to_data_url(gen_video_bytes, mime_type="video/mp4"),
                },
            })

        # GT frames
        if subtrack == Subtrack.DEDUCTION:
            if gt_frames:
                content.append({
                    "type": "text",
                    "text": f"GROUND TRUTH FRAMES ({len(gt_frames)} sampled):",
                })
                for frame in gt_frames[: self.opt_config.num_eval_frames]:
                    jpeg = frame_to_jpeg_bytes(frame)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": bytes_to_data_url(jpeg)},
                    })

        messages = [{"role": "user", "content": content}]
        raw_response = self.client.chat_completion(messages)
        return self._parse_scores(raw_response, subtrack)

    def _parse_scores(self, raw: str, subtrack: Subtrack) -> SubtrackScores:
        """Parse LLM JSON response into SubtrackScores."""
        data = parse_llm_json(raw)
        if data is None:
            logger.error(f"Failed to parse JSON: {raw[:300]}")
            scores = SubtrackScores()
            scores.feedback = f"PARSE_ERROR: {raw[:500]}"
            return scores

        def sf(key: str) -> float:
            v = data.get(key, 0.0)
            try:
                val = float(v)
                if val < 0.0:
                    return -1.0
                return max(0.0, min(1.0, val))
            except (ValueError, TypeError):
                return 0.0

        scores = SubtrackScores()
        for field_name in SUBTRACK_FIELDS[subtrack]:
            setattr(scores, field_name, sf(field_name))
        scores.feedback = data.get("feedback", "")
        return scores
