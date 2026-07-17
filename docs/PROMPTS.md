# Inference prompts

The exact version-1.0 paper prompts are stored in `apple_pi/prompts/templates.py`:

- `VM_PROMPT_TEMPLATES`: video-model protocol.
- `UM_PROMPT_TEMPLATES`: image-model protocol.

Runtime fields are filled from case metadata:

- `formula_choices` for `formulation_text`;
- `target_time` for `formulation_graphic`;
- `physics_duration` for video `deduction`;
- `time_point` for image `deduction`.

Use `apple-pi export-prompts` to avoid manually formatting these values. Three rollouts use the same released prompt and independent model sampling.
