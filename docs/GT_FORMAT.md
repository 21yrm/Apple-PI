# Apple-PI GT format

The complete Apple-π v1.0 GT dataset, **Orchard**, is publicly available at
[yaorunmao/Apple-PI-GT](https://huggingface.co/datasets/yaorunmao/Apple-PI-GT).
It contains all 400 test cases. This guide describes the directory layout and
metadata conventions of the published dataset.

## Dataset root

The published v1.0 repository has the following logical layout:

```text
Apple-PI-GT/
├── .gitattributes
├── README.md
├── assets/                              # dataset-card media
├── dataset.json
└── cases/
    ├── UniversalGravitation/
    │   ├── FreeFall/
    │   ├── ProjectileMotion/
    │   ├── InclinedPlane/
    │   └── CircularMotion/
    ├── ConservationOfMomentum/
    │   ├── PerfectlyElasticCollision/
    │   ├── PerfectlyInelasticCollision/
    │   └── InelasticCollision/
    ├── NewtonsFirstLaw/
    │   ├── AtRest/
    │   └── UniformLinearMotion/
    └── MultiLaw/
        └── Composition/
```

The dataset-card files (`README.md` and `assets/`) and `.gitattributes` are
repository support files. The evaluator reads `dataset.json` and the case
directories under `cases/`.

The 400 entries in the published manifest are distributed as follows:

| Law pillar | Task | Cases |
|---|---|---:|
| `UniversalGravitation` | `FreeFall` | 61 |
| `UniversalGravitation` | `ProjectileMotion` | 90 |
| `UniversalGravitation` | `InclinedPlane` | 24 |
| `UniversalGravitation` | `CircularMotion` | 26 |
| `ConservationOfMomentum` | `PerfectlyElasticCollision` | 41 |
| `ConservationOfMomentum` | `PerfectlyInelasticCollision` | 35 |
| `ConservationOfMomentum` | `InelasticCollision` | 41 |
| `NewtonsFirstLaw` | `AtRest` | 36 |
| `NewtonsFirstLaw` | `UniformLinearMotion` | 25 |
| `MultiLaw` | `Composition` | 21 |
| **Total** |  | **400** |

Directory names are PascalCase versions of the law pillars and task names in
the paper. Punctuation is omitted: “Newton's first law” is
`NewtonsFirstLaw`, and “multi-law” is `MultiLaw`. The released gravitational
pillar is named `UniversalGravitation`, not `Gravity`.

Every case ID has the form
`<LawPillar>/<Task>/<six-digit case number>`. The root-level `dataset.json`
indexes those IDs and their corresponding directories:

```json
{
  "name": "Apple-PI",
  "version": "1.0",
  "prompt_version": "1.0",
  "num_rollouts": 3,
  "cases": [
    {
      "case_id": "UniversalGravitation/FreeFall/000000",
      "path": "cases/UniversalGravitation/FreeFall/000000",
      "split": "test"
    }
  ]
}
```

`case_id` is a unique POSIX-style relative path without the `cases/` prefix;
`path` is relative to the dataset root and includes that prefix.

## Case directories

Every published case contains the common assets below:

```text
<case directory>/
├── metadata.json
├── initial_state/
├── instantaneous_velocity/
├── rgb/
├── instance_segmentation/
└── mask/
```

Simulator cases (`metadata.json: is_realworld` is `false`) additionally
contain dense geometry and camera data:

```text
<simulator case directory>/
├── depth/
├── velocity/
└── camera_parameters/
```

These three simulator-only directories are absent from real-world cases.

## Case metadata

Each case directory contains a `metadata.json` file. This representative
record is from `UniversalGravitation/FreeFall/000000` in the published
dataset:

```json
{
  "schema_version": "1.0",
  "case_id": "UniversalGravitation/FreeFall/000000",
  "physics_type": "FreeFall",
  "is_realworld": false,
  "annotation": "Environment: g=-9.81, F_drag=0.0\nGround: e = 0.4\nClaySphere: d = 1.0, e = 0.2, h = 6.0, v_0 = 0.0",
  "input_image": "initial_state/rgb_0000.png",
  "clean_first_frame": "rgb/0000.png",
  "annotations_only_reference": "initial_state/rgb_0000_white_bg.png",
  "objects_only_reference": "initial_state/rgb_0000_white_bg_obj.png",
  "future_state_reference": "instantaneous_velocity/velocity_annotated.png",
  "target_time": 0.5,
  "physics_duration": 1,
  "gt_frames_dir": "rgb",
  "gt_video": null,
  "gt_fps": 24.0,
  "gt_total_frames": 241,
  "deduction_timestamps": [0.5, 1.0],
  "formula_info": {
    "choices": [
      "n_1·sinθ_1 = n_2·sinθ_2",
      "y(t) = h + v_0·t + (1/2)·g·t²",
      "F_fall = m·g·(1 + v/c) / √(1 - v²/c²)",
      "P + (1/2)·ρ·v² + ρ·g·h = constant"
    ],
    "correct_letter": "B",
    "correct_formula": "y(t) = h + v_0·t + (1/2)·g·t²"
  }
}
```

Paths are relative to the case directory.

## Condition-frame convention

Apple-PI interprets frame indices as follows:

- Dense GT array index `0` and GT video frame `0` are the condition frame.
- Physics frames begin at index `1`.
- `rgb/0000.png` is the clean condition frame.
- For Deduction, frame `0` of each prediction video is also the condition frame.

Every dense GT array includes the condition frame at index `0`.

## Visual assets

```text
initial_state/
├── rgb_0000.png
├── rgb_0000_white_bg.png
├── rgb_0000_white_bg_obj.png
├── mask_0000.npy
├── instance_segmentation_0000.npy
└── instance_segmentation_mapping_0000.json # simulator cases

instantaneous_velocity/
├── velocity_annotated.png
├── mask.npy
└── mapping.json                            # simulator cases
```

For real-world cases, Apple-PI treats every non-zero instance ID as an object.
The simulator-only mapping JSON files are not required.

## GT RGB

Published v1.0 simulator cases store GT RGB frames as a PNG sequence and set
`gt_video` to `null`:

```text
rgb/
├── 0000.png
├── 0001.png
├── 0002.png
└── ...
```

Published v1.0 real-world cases retain a standalone copy of the condition
frame as `0000.png` for direct image access. The video referenced by
`gt_video` is the complete GT sequence: video frame `0` is the same condition
frame, and video frames `1` onward are the physics frames.

```text
rgb/
├── 0000.png
└── video.mp4
```

When `gt_video` is non-null, consumers should decode that video directly and
must not prepend `rgb/0000.png`, which would duplicate the condition frame.
Consumers should follow the paths declared in `metadata.json` rather than
inferring the source type from filenames alone.

## Programmatic metric arrays

Each NPZ file below stores its array under the key `maps`.

Expected shapes:

| Asset | Cases | Shape | Meaning |
|---|---|---|---|
| `depth/maps.npz` | Simulator only | `[T, H, W]` | GT depth including condition frame |
| `instance_segmentation/maps.npz` | All | `[T, H, W]` | Integer instance IDs including condition frame |
| `mask/maps.npz` | All | `[T, H, W]` | Binary/foreground masks including condition frame |
| `velocity/maps.npz` | Simulator only | `[T, H, W, 3]` | Dense world-space velocity including condition frame |

The evaluator uses the stored instance IDs and mappings directly, so instance
IDs remain consistent across frames.

## Camera parameters

Deduction metrics for simulator cases use per-frame camera parameters:

```text
camera_parameters/
├── 0000.json
├── 0001.json
└── ...
```

Each JSON file contains:

```json
{
  "intrinsic": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
  "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
}
```

`intrinsic` has shape `[3, 3]`, and `extrinsic` has shape `[3, 4]`. The evaluator reads the rotation from `extrinsic[:3, :3]` and the translation from `extrinsic[:3, 3]`. These matrices use the same coordinate convention as the associated depth and velocity data because the MoGe and velocity metrics consume them directly.

## Validation

To check a local GT directory, run:

```bash
apple-pi validate-gt --gt-dir /path/to/Apple-PI-GT
```
