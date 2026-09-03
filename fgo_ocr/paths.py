from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def _data_root() -> Path:
    env = (os.environ.get("FGO_OCR_DATA") or os.environ.get("FGO_OCR_OUT") or "").strip()
    if env:
        return Path(env)
    if os.name == "nt" and Path("D:/").exists():
        return Path(r"D:\FGO-OCR-data")
    if Path("/mnt/d").exists():
        return Path("/mnt/d/FGO-OCR-data")
    return ROOT / "data"


DATA = _data_root()
MODELS = ROOT / "models"
LABELS = ASSETS / "labels.txt"
ATLAS_QUESTS = ASSETS / "atlas_quests.txt"
ATLAS_NAMES = ASSETS / "atlas_names.txt"
CHARSET = ASSETS / "charset.txt"