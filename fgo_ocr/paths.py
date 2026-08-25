from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
MODELS = ROOT / "models"
LABELS = ASSETS / "labels.txt"
ATLAS_QUESTS = ASSETS / "atlas_quests.txt"
ATLAS_NAMES = ASSETS / "atlas_names.txt"
CHARSET = ASSETS / "charset.txt"
CHARSET_ATLAS = ASSETS / "charset_atlas.txt"