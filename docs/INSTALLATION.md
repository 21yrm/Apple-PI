# Reproducible installation

This is the supported clean-machine installation path for the full Apple-PI
evaluation stack. It installs the CLI, Gemini SDK, SAM3 integration, MoGe-2,
and the pinned CUDA-enabled PyTorch build.

## 0. System requirements

- Linux x86-64 with an NVIDIA GPU. The full evaluator uses BF16 SAM3 and MoGe-2;
  a CUDA GPU with at least 24 GB of VRAM is recommended.
- An NVIDIA driver compatible with the official PyTorch CUDA 12.8 wheel. Check
  that `nvidia-smi` works before creating the environment.
- Conda or Miniforge. `git` and `ffmpeg` are installed by `environment.yml`.
- At least 25 GB of free disk space for the environment, source dependencies,
  and model cache. The two evaluation checkpoints occupy about 4.6 GB.
- Network access to GitHub, Hugging Face, and the Google Gemini API.

Prompt export and data validation can run on CPU. The full paper evaluation is
the GPU workflow described here.

## 1. Clone and create the environment

```bash
git clone https://github.com/21yrm/Apple-PI.git
cd Apple-PI
conda env create -f environment.yml
conda activate apple-pi
```

The environment uses Python 3.10 and the official PyTorch 2.10 CUDA 12.8
wheels. PyTorch stopped publishing official conda packages after PyTorch 2.5,
so PyTorch is intentionally installed by pip inside the conda environment.

No Git submodules or manual third-party clones are required. Installation pulls
the pinned MoGe source revision automatically. MoGe's own pinned `utils3d`
and `pipeline` dependencies are also installed automatically by pip.

The complete pinned Git-source dependency chain is:

| Package | Source revision |
|---|---|
| MoGe | `microsoft/MoGe@07444410f1e33f402353b99d6ccd26bd31e469e8` |
| utils3d | `EasternJournalist/utils3d@3fab839f0be9931dac7c8488eb0e1600c236e183` |
| pipeline | `EasternJournalist/pipeline@866f059d2a05cde05e4a52211ec5051fd5f276d6` |

The latter two pins come from the fixed MoGe revision. They are listed here so
users can audit the resolver output; they do not need to clone them manually.

For an existing Python 3.10 environment, the equivalent fallback is:

```bash
python -m pip install -r requirements.txt
```

The conda workflow remains the supported path because it also provides
`ffmpeg`, `git`, and a predictable Python version.

## 2. Configure external access

### Hugging Face / SAM3

1. Request and accept access to
   [`facebook/sam3`](https://huggingface.co/facebook/sam3).
2. Authenticate in the same user account and environment that will run the
   evaluation:

```bash
hf auth login
```

### Google Gemini

Create an official Gemini API key in
[Google AI Studio](https://aistudio.google.com/app/apikey), then either export
it for the current shell:

```bash
export GEMINI_API_KEY="your-key"
```

or use the git-ignored local environment file:

```bash
cp .env.example .env
chmod 600 .env
# Edit .env and set GEMINI_API_KEY without committing the file.
```

The evaluator sends generated images/videos and the reference images required
by each Gemini rubric to Google's Gemini API. Do not evaluate private artifacts
unless that data transfer is permitted by your organization or dataset terms.

## 3. Pre-download the pinned checkpoints

Run this once before a long evaluation:

```bash
apple-pi download-models
```

It downloads exactly these pinned snapshots into the Hugging Face cache:

| Component | Repository | Pinned revision | Approximate size |
|---|---|---|---:|
| SAM3 | `facebook/sam3` | `3c879f39826c281e95690f02c7821c4de09afae7` | 3.3 GB |
| MoGe-2 normal | `Ruicheng/moge-2-vitl-normal` | `b135031bae30b5ac2ae141a0e68717795ce38340` | 1.3 GB |

Use `HF_HOME=/path/with/more/space` before this command if the default cache
location does not have enough disk space. Evaluation uses the same cache.

## 4. Verify the installation

```bash
apple-pi doctor
```

A full evaluation environment should report `OK` for PyTorch, CUDA, SAM3
Transformers classes, MoGe-2, `GEMINI_API_KEY`, and the Hugging Face token.

The critical package versions for this release are:

| Package | Version/source |
|---|---|
| PyTorch | `2.10.0` (CUDA 12.8 wheel) |
| torchvision | `0.25.0` |
| Transformers | `5.1.0` |
| google-genai | `1.60.0` |
| MoGe | commit `07444410f1e33f402353b99d6ccd26bd31e469e8` |

## 5. Download data and run

```bash
apple-pi download-data --output data/apple_pi
apple-pi validate-gt --gt-dir data/apple_pi
apple-pi validate-predictions \
  --gt-dir data/apple_pi \
  --pred-dir predictions/my_model
apple-pi evaluate \
  --gt-dir data/apple_pi \
  --pred-dir predictions/my_model \
  --output results/my_model.json
```

The first two validation commands do not call Gemini or run SAM3/MoGe. Always
run them before spending API or GPU compute.
