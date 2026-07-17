# Prediction preparation

Apple-PI evaluates exactly three independent rollouts for each case and subtrack. The evaluator does not call a third-party generation model and does not repair semantic misalignment.

## Submission metadata

```jsonc
{
  "model": "model-name",
  "protocol": "video", // Use "image" for image models.
  "num_rollouts": 3,
  "prompt_version": "1.0",
  "notes": "optional generation settings"
}
```

## Video protocol

Each subtrack directory contains `rollout_00.mp4`, `rollout_01.mp4`, and `rollout_02.mp4`.

For the first four subtracks, the last decodable frame is evaluated.

For `deduction`, frame 0 is skipped as the condition frame. Remaining frames are interpreted as covering the case's full `physics_duration`. The evaluator derives the prediction timeline from `physics_duration` and the number of physics frames, so you do not need to normalize the MP4 container FPS. It samples prediction frames at GT timestamps and resizes them to GT spatial resolution. The evaluator does not detect or crop solid-color border padding (e.g., black, white, or gray); such padding must be removed before submission.

## Image protocol

The first four subtracks contain three PNG rollouts.

Deduction contains one directory per rollout. Required timestamps come from the case metadata. Canonical filename format:

```text
frame_000_t0.500s.png
frame_001_t1.000s.png
```

Every timestamp is independently generated from the original condition image with the released image-model prompt.

## Validation

```bash
apple-pi validate-predictions \
  --gt-dir data/apple_pi \
  --pred-dir predictions/your_model
```

Run this before evaluation. Missing rollouts, missing keyframes, unreadable images, and unreadable videos are reported without invoking Gemini, SAM3, or MoGe.
