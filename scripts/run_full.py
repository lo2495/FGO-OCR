from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _cpus() -> int:
    return max(1, os.cpu_count() or 8)


def _boot() -> Path:
    data = os.environ.get("FGO_OCR_DATA") or os.environ.get("FGO_OCR_OUT") or r"D:\FGO-OCR-full"
    if not Path("D:/").exists() and not Path(data).exists() and str(data).lower().startswith("d:\\"):
        data = str(ROOT / "data" / "full")
        print(f"D: 不存在，改用 {data}", flush=True)
    os.environ["FGO_OCR_DATA"] = data
    os.environ["FGO_OCR_OUT"] = data
    os.environ.setdefault("FGO_OCR_MIX", "full")
    os.environ.setdefault("FGO_OCR_N", "600000")
    os.environ.setdefault("FGO_OCR_EPOCHS", "180")
    os.environ.setdefault("FGO_OCR_LR", "8e-5")
    os.environ.setdefault("FGO_OCR_RESUME", "1")
    os.environ.setdefault("FGO_OCR_WORKERS", str(min(8, _cpus())))
    os.environ.setdefault("FGO_OCR_SYNTH_WORKERS", str(max(1, _cpus() - 1)))
    Path(data).mkdir(parents=True, exist_ok=True)
    print(
        f"full boot data={data} n={os.environ['FGO_OCR_N']} "
        f"epochs={os.environ['FGO_OCR_EPOCHS']} lr={os.environ['FGO_OCR_LR']} "
        f"resume={os.environ['FGO_OCR_RESUME']} mix={os.environ['FGO_OCR_MIX']} "
        f"synth_workers={os.environ['FGO_OCR_SYNTH_WORKERS']} "
        f"dl_workers={os.environ['FGO_OCR_WORKERS']}",
        flush=True,
    )
    return Path(data)


if __name__ == "__main__":
    _boot()
    from fgo_ocr.atlas import main as atlas_main
    from fgo_ocr.synth import main as synth_main
    from fgo_ocr.train_hires import main as train_main

    print("===== atlas =====", flush=True)
    atlas_main()
    print("===== synth full =====", flush=True)
    synth_main()
    print("===== train_hires =====", flush=True)
    train_main()
    print("===== done =====", flush=True)
