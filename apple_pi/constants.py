"""Public release constants."""

BENCHMARK_NAME = "Apple-PI"
BENCHMARK_VERSION = "1.0"
PROMPT_VERSION = "1.0"
NUM_ROLLOUTS = 3

DEFAULT_DATASET_REPO = "yaorunmao/Apple-PI-GT"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# Revisions used by the reproducible public evaluation environment. Pinning model
# snapshots keeps future upstream changes from silently changing benchmark
# results or breaking the documented installation.
SAM3_MODEL_REPO = "facebook/sam3"
SAM3_MODEL_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
MOGE_MODEL_REPO = "Ruicheng/moge-2-vitl-normal"
MOGE_MODEL_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
