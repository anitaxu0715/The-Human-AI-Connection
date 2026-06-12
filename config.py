"""
Central configuration for the Human-AI Connection research pipeline.

Override the data directory at runtime:
  python script.py --data-dir /path/to/data
  DATA_DIR=/path/to/data python script.py
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────

# Raw .zst archives live here. Override via --data-dir or DATA_DIR env var.
DEFAULT_DATA_DIR = r"D:\Apps\dataset\reddit\subreddits25"

# Output subdirectories (relative to this file's location)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR  = os.path.join(OUTPUTS_DIR,  "figures")
SAMPLES_DIR  = os.path.join(OUTPUTS_DIR,  "samples")


def get_data_dir() -> str:
    """Return data directory from env var, falling back to the default."""
    return os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)


def data_path(filename: str, data_dir: str | None = None) -> str:
    """Build a full path inside the data directory."""
    return os.path.join(data_dir or get_data_dir(), filename)


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for d in (OUTPUTS_DIR, FIGURES_DIR, SAMPLES_DIR):
        os.makedirs(d, exist_ok=True)


# ── Subreddits ────────────────────────────────────────────────────────────────

TARGET_SUBREDDITS = {
    "ethics", "replika", "characterai", "kindroidai", "soulmateai",
    "localllm", "singularity", "socialanxiety", "nomiai", "artificialinteligence",
}

# Subreddits small enough to keep all rows (no sampling cap applied)
FULL_SAMPLE_SUBREDDITS = {"ethics", "kindroidai", "soulmateai", "nomiai"}

# Maximum rows drawn from each large subreddit
LARGE_SUB_SAMPLE = 100_000

# ── Processing ────────────────────────────────────────────────────────────────

CHUNKSIZE        = 200_000   # pandas rows per chunk
WRITE_BUFFER     = 5_000     # rows buffered before flushing to CSV
RANDOM_SEED      = 42        # for future extensions that use random sampling

# ── VADER thresholds ──────────────────────────────────────────────────────────

POS_THRESHOLD =  0.05
NEG_THRESHOLD = -0.05
