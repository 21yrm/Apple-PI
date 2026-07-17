<h1 align="center">Apple-π: Benchmarking Thinking with Video<br>Towards Law-Grounded Physical Intelligence</h1>

<p align="center"><strong>Runmao Yao<sup>*</sup>, Kairui Hu<sup>*</sup>, Yukang Cao, Ruisi Wang, Shulin Tian, Ziang Cao, Weichen Fan,<br>Ziqi Huang, Yuhao Dong, Hao Li, Zhaoxi Chen, Zhongang Cai, Lei Yang, Ziwei Liu<sup>†</sup></strong></p>

<p align="center">S-Lab, Nanyang Technological University · The Chinese University of Hong Kong<br><sup>*</sup>Equal contribution · <sup>†</sup>Corresponding author</p>

<p align="center">
  <a href="https://21yrm.github.io/Apple-PI-homepage/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Project-Page-2ea44f?logo=apple&amp;logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/papers/2607.16401"><img src="https://img.shields.io/badge/HF-Paper-FFD21E?logo=huggingface&amp;logoColor=white" alt="HF Paper"></a>
  <a href="https://arxiv.org/abs/2607.16401"><img src="https://img.shields.io/badge/arXiv-2607.16401-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv 2607.16401"></a>
  <a href="https://www.youtube.com/watch?v=IduINd9phxw"><img src="https://img.shields.io/badge/YouTube-Video-ff0000?logo=youtube&amp;logoColor=white" alt="YouTube Video"></a>
  <a href="https://huggingface.co/datasets/yaorunmao/Apple-PI-GT"><img src="https://img.shields.io/badge/HF-Dataset-FFD21E?logo=huggingface&amp;logoColor=white" alt="HF Dataset"></a>
  <a href="https://huggingface.co/spaces/yaorunmao/apple-pi-leaderboard"><img src="https://img.shields.io/badge/HF-Leaderboard-FFD21E?logo=huggingface&amp;logoColor=white" alt="HF Leaderboard"></a>
</p>

<p align="center">
  <img src="assets/apple_pi_teaser.png" width="100%" alt="Apple-π benchmark overview">
</p>

## 📰 News

- **2026/08:** The complete Apple-π GT dataset is released on [Hugging Face](https://huggingface.co/datasets/yaorunmao/Apple-PI-GT).
- **2026/07:** Project page, public leaderboard, and demo video are online.
- **2026/07:** Inference prompts and paper-compatible evaluation code prepared for release.

## 📖 Overview

Modern video models can produce motion that looks plausible, but visual plausibility alone does not show that a model has identified the relevant quantities, selected the governing physical law, and followed that law through time. **Apple-π** is a diagnostic benchmark that anchors video-model evaluation explicitly in physical laws and makes the model’s reasoning process visible through generated frames.

Apple-π contains three coupled components:

- **Orchard:** 400 videos covering 10 canonical classical-mechanics tasks, with simulated, self-recorded, and Internet-sourced cases. The taxonomy separates nine single-law tasks for controlled diagnosis from a multi-law composition task for generalization.
- **Benchmark protocol:** a scientific-reasoning pipeline organized as **Perception → Formulation → Deduction**. Perception and Formulation each contain text and graphic subtracks, producing five subtracks in total.
- **Evaluation suite:** Gemini-based subjective judging together with physics-law-grounded objective metrics, including SAM3 segmentation/tracking, MoGe depth estimation, mask-based image metrics, trajectory overlap, and velocity error.

The paper evaluates video models through **chain-of-frames** outputs. This code release also publishes an image-model protocol: image models produce final-frame artifacts for the first four subtracks and independently generated timestamped keyframes for Deduction.

Every case/subtrack is evaluated with **three independent rollouts**. The release preserves the paper implementation of prompt wording, score definitions and weights, Gemini judging, SAM3/MoGe behavior, temporal handling, and programmatic metrics.

> **GT dataset:** the complete Apple-π v1.0 ground-truth dataset, **Orchard**, is publicly available at [yaorunmao/Apple-PI-GT](https://huggingface.co/datasets/yaorunmao/Apple-PI-GT). It contains all 400 test cases and the annotations required by the evaluator. The paths, schemas, and commands below describe this published release.

## 🧩 Benchmark subtracks

| Subtrack | Question | Expected output |
|---|---|---|
| Perception-Text | Read physical quantities | Final white-background annotation artifact |
| Perception-Graphic | Ground physical objects | Final white-background object-only artifact |
| Formulation-Text | Select governing law | Final white-background three-line formula answer |
| Formulation-Graphic | Predict state at target time | Scene at target time with velocity arrows and labels |
| Deduction | Generate full dynamics | Full trajectory video or timestamped keyframes |

The exact paper prompts are versioned in [`apple_pi/prompts/templates.py`](apple_pi/prompts/templates.py). The CLI only fills case-specific placeholders; it does not rewrite prompt wording.

## ⚙️ Installation

The official evaluation includes Gemini 3.0 Flash, gated SAM3, and MoGe-2. The
full evaluator requires an NVIDIA CUDA GPU with at least 24 GB of VRAM
recommended.
See the [reproducible installation guide](docs/INSTALLATION.md) for system
requirements, pinned versions/checkpoints, cache size, and troubleshooting.

```bash
git clone https://github.com/21yrm/Apple-PI.git
cd Apple-PI

conda env create -f environment.yml
conda activate apple-pi
```

No Git submodules or manual third-party clones are required. The environment
installs the pinned MoGe commit and its dependencies automatically.

SAM3 uses the gated [`facebook/sam3`](https://huggingface.co/facebook/sam3)
checkpoint. Accept access and authenticate in the same user account that will
run evaluation:

```bash
hf auth login
```

Set an official Google AI Studio Gemini key:

```bash
export GEMINI_API_KEY="your-key"
```

Pre-download the pinned SAM3 and MoGe-2 checkpoints (about 4.6 GB total), then
verify the complete environment:

```bash
apple-pi download-models
apple-pi doctor
```

The `environment.yml` pins the API-sensitive package versions and model source
revision. Model snapshot revisions are pinned in the code and listed in
[`docs/INSTALLATION.md`](docs/INSTALLATION.md).

The paper judge is **Gemini 3.0 Flash**. The public Gemini API model code used by this release is `gemini-3-flash-preview`. Google documents image/video input through the [Gemini API](https://ai.google.dev/gemini-api/docs/video-understanding) and the model code on the [Gemini 3 Flash model page](https://ai.google.dev/gemini-api/docs/models/gemini-3-flash-preview).

## 📥 1. Download GT data

Download the complete GT dataset:

```bash
apple-pi download-data --output data/apple_pi
```

Equivalent Hugging Face command:

```bash
hf download \
  yaorunmao/Apple-PI-GT \
  --repo-type dataset \
  --local-dir data/apple_pi
```

Validate the downloaded data before use:

```bash
apple-pi validate-gt --gt-dir data/apple_pi
```

Published cases use the path
`cases/<LawPillar>/<Task>/<six-digit case number>/`. For example, case ID
`UniversalGravitation/FreeFall/000000` maps to
`cases/UniversalGravitation/FreeFall/000000`. The four released law pillars
are `UniversalGravitation`, `ConservationOfMomentum`, `NewtonsFirstLaw`, and
`MultiLaw`.

See [docs/GT_FORMAT.md](docs/GT_FORMAT.md) for the complete 10-task directory
layout, per-case assets, and metadata conventions of the published dataset.
Machine-readable schemas are provided in [`schemas/`](schemas/).

## 📝 2. Export inference prompts

Video-model prompts:

```bash
apple-pi export-prompts \
  --gt-dir data/apple_pi \
  --protocol video \
  --output prompts/video.jsonl
```

Image-model prompts:

```bash
apple-pi export-prompts \
  --gt-dir data/apple_pi \
  --protocol image \
  --output prompts/image.jsonl
```

Each JSONL record contains the case ID, subtrack, input image, rendered prompt, expected output type, prompt version, and required rollout count.

## 🎬 3. Run your model three times

Generate **three independent rollouts** for every case and subtrack. Do not select the best sample. Store results using the structure below.

### 🎥 Video-model submission

```text
predictions/my_video_model/
├── submission.json
└── cases/
    └── UniversalGravitation/FreeFall/000000/
        ├── perception_text/
        │   ├── rollout_00.mp4
        │   ├── rollout_01.mp4
        │   └── rollout_02.mp4
        ├── perception_graphic/
        ├── formulation_text/
        ├── formulation_graphic/
        └── deduction/
            ├── rollout_00.mp4
            ├── rollout_01.mp4
            └── rollout_02.mp4
```

### 🖼️ Image-model submission

```text
predictions/my_image_model/
├── submission.json
└── cases/
    └── UniversalGravitation/FreeFall/000000/
        ├── perception_text/
        │   ├── rollout_00.png
        │   ├── rollout_01.png
        │   └── rollout_02.png
        ├── perception_graphic/
        ├── formulation_text/
        ├── formulation_graphic/
        └── deduction/
            ├── rollout_00/
            │   ├── frame_000_t0.500s.png
            │   ├── frame_001_t1.000s.png
            │   └── ...
            ├── rollout_01/
            └── rollout_02/
```

`submission.json`:

```jsonc
{
  "model": "your-model-name",
  "protocol": "video", // Use "image" for image models.
  "num_rollouts": 3,
  "prompt_version": "1.0",
  "notes": ""
}
```

## 🛠️ 4. Prepare predictions for evaluation

Apple-PI does **not** preprocess or semantically align third-party outputs. You must prepare your results before running the evaluator.

For `perception_text`, `perception_graphic`, `formulation_text`, and `formulation_graphic`:

- Video evaluation uses the last decodable frame.
- Put the final answer in the last frame.
- Remove provider watermarks, title cards, trailing black frames, and unrelated outro frames before evaluation.

For video `deduction`:

- Frame 0 must be the conditioning/initial-state frame.
- All following frames are treated as physics frames.
- The evaluator interprets those physics frames as covering `physics_duration` from the case metadata and samples them at GT timestamps.
- Do not add intro/outro frames or static padding.
- The evaluator resizes prediction frames to GT resolution when required, but it does not discover or repair semantic time offsets.

For image `deduction`:

- Generate every timestamp listed in `metadata.json:deduction_timestamps`.
- Each timestamp is generated independently from the original input image using the released prompt.
- Filenames must use `frame_{index:03d}_t{time:.3f}s.png`.

See [docs/PREDICTION_FORMAT.md](docs/PREDICTION_FORMAT.md) for examples and preparation guidance.

## 📊 5. Validate and evaluate

Validate all three rollouts before spending Gemini/GPU compute:

```bash
apple-pi validate-predictions \
  --gt-dir data/apple_pi \
  --pred-dir predictions/my_video_model
```

Run the complete paper-compatible evaluation:

```bash
apple-pi evaluate \
  --gt-dir data/apple_pi \
  --pred-dir predictions/my_video_model \
  --output results/my_video_model.json
```

The command runs the paper implementation of:

- Gemini 3.0 Flash judging;
- SAM3 segmentation and tracking;
- MoGe depth estimation;
- the existing programmatic metrics and score fusion;
- three-rollout aggregation.

Gemini responses are cached under `.cache/apple_pi/gemini`, so interrupted runs do not pay for identical completed requests again.

A single subtrack can be evaluated with:

```bash
apple-pi evaluate \
  --gt-dir data/apple_pi \
  --pred-dir predictions/my_video_model \
  --subtrack deduction \
  --output results/my_video_model_deduction.json
```

## 🗂️ Repository structure

```text
.
├── apple_pi/
│   ├── prompts/                 # exact video/image paper prompts
│   ├── data/                    # GT and prediction contracts
│   ├── evaluation/
│   │   ├── paper_evaluator.py   # Gemini prompts and score definitions
│   │   ├── segmentation.py      # SAM3 single-image evaluation
│   │   ├── programmatic_metrics.py
│   │   ├── runners.py           # video/image protocol runners
│   │   └── submission.py        # three-rollout orchestration
│   └── cli/
├── docs/                        # public data and submission protocols
├── schemas/                     # machine-readable contracts
└── assets/                      # benchmark teaser
```

## 📚 Citation

```bibtex
@misc{yao2026applepibenchmarkingthinkingvideo,
      title={Apple-$\pi$: Benchmarking Thinking with Video Towards Law-Grounded Physical Intelligence},
      author={Runmao Yao and Kairui Hu and Yukang Cao and Ruisi Wang and Shulin Tian and Ziang Cao and Weichen Fan and Ziqi Huang and Yuhao Dong and Hao Li and Zhaoxi Chen and Zhongang Cai and Lei Yang and Ziwei Liu},
      year={2026},
      eprint={2607.16401},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.16401},
}
```

## ⚖️ License

The evaluation code is released under the Apache License 2.0; see [LICENSE](LICENSE).
The [Orchard GT dataset](https://huggingface.co/datasets/yaorunmao/Apple-PI-GT)
is released under CC BY 4.0 unless a file or source attribution states
otherwise. See the dataset card for source-specific attribution and usage
requirements.
