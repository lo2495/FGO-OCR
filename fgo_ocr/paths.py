from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
MODELS = ROOT / "models"
LABELS = ASSETS / "labels.txt"
CHARSET = ASSETS / "charset.txt"
