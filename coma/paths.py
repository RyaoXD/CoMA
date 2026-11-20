"""Repository path helpers.

Override with environment variables when packaging or deploying:

- ``COMA_DATA_DIR``: root directory that contains MIMIC-CXR / CheXpert / RSNA / ...
- ``COMA_OUTPUT_DIR``: checkpoints, TensorBoard / WandB logs, and other run artifacts
"""

from __future__ import annotations

import os
from pathlib import Path

# coma/paths.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_data_dir() -> Path:
    """Resolve a portable dataset root.

    Priority:
    1. ``COMA_DATA_DIR``
    2. ``<repo>/data`` if it already contains known datasets
    3. Legacy layout ``<repo>/../../dataset`` (e.g. ``/mnt/E/dataset``)
    4. Fallback to ``<repo>/data`` (create / populate via preprocess)
    """
    if "COMA_DATA_DIR" in os.environ:
        return Path(os.environ["COMA_DATA_DIR"]).expanduser().resolve()

    repo_data = REPO_ROOT / "data"
    markers = (
        "MIMIC-CXR-JPG",
        "CheXpert-v1.0",
        "RSNA",
        "SIIM",
        "COVIDx",
        "object-CXR",
    )
    if any((repo_data / name).exists() for name in markers):
        return repo_data.resolve()

    legacy = REPO_ROOT.parent.parent / "dataset"
    if legacy.exists():
        return legacy.resolve()

    return repo_data.resolve()


DATA_BASE_DIR = _default_data_dir()

OUTPUT_DIR = Path(
    os.environ.get("COMA_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
).expanduser().resolve()

CKPT_DIR = OUTPUT_DIR / "coma_ckpts"
LOG_DIR = OUTPUT_DIR / "coma_data"
