from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _boot() -> Path:
    data = os.environ.get("FGO_OCR_DATA") or os.environ.get("FGO_OCR_OUT") or r"D:\FGO-OCR-data"
    if not Path("D:/").exists() and not Path(data).exists() and data.lower().startswith("d:\\"):
        data = str(ROOT / "data")
        print(f"D: 不存在，改用 {data}", flush=True)
    os.environ["FGO_OCR_DATA"] = data
    os.environ["FGO_OCR_OUT"] = data
    os.environ.setdefault("FGO_OCR_N", "160000")
    os.environ.setdefault("FGO_OCR_EPOCHS", "120")
    os.environ.setdefault("FGO_OCR_LR", "3e-4")
    os.environ.setdefault("FGO_OCR_RESUME", "1")
    os.environ.setdefault("FGO_OCR_WORKERS", "4")
    Path(data).mkdir(parents=True, exist_ok=True)
    print(
        f"hires boot data={data} n={os.environ['FGO_OCR_N']} "
        f"epochs={os.environ['FGO_OCR_EPOCHS']} resume={os.environ['FGO_OCR_RESUME']}",
        flush=True,
    )
    return Path(data)


if __name__ == "__main__":
    _boot()
    from fgo_ocr.synth import main as synth_main
    from fgo_ocr.train_hires import main as train_main

    print("===== synth =====", flush=True)
    synth_main()
    print("===== train_hires =====", flush=True)
    train_main()
    print("===== done =====", flush=True)
