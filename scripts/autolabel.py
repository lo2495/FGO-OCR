from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["OCR_BACKEND"] = "sarashina"

import cv2

AGENT = Path(os.environ.get("FGO_AGENT_ROOT", r"F:\MyOwnProject\FGO-Vision-Agent"))
OCR_ROOT = Path(__file__).resolve().parents[1]
REAL = Path(os.environ.get("FGO_OCR_REAL", str(OCR_ROOT / "data" / "real")))

sys.path.insert(0, str(AGENT))
os.chdir(AGENT)

from core.ocr_engine import OCREngine


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if folder is None:
        print("指定資料夾（只標該目錄下「空 txt」）：")
        print("  python autolabel.py data/real/tab")
        print("已有文字的 txt 不會覆寫。")
        return
    root = folder if folder.is_absolute() else (OCR_ROOT / folder)
    pngs = sorted(root.glob("*.png"))
    print(f"root={root} png={len(pngs)} backend=sarashina", flush=True)
    if not pngs:
        raise SystemExit("沒有 png")
    ocr = OCREngine()
    ocr.wait_ready(timeout=180.0)
    n = skip = 0
    for p in pngs:
        txt = p.with_suffix(".txt")
        if txt.is_file() and txt.read_text(encoding="utf-8").strip():
            skip += 1
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        text = (ocr.read_text(img) or "").strip().replace("\n", " ")
        txt.write_text(text + "\n", encoding="utf-8")
        n += 1
        print(f"{n} {p.name}  {text}", flush=True)
    print(f"wrote={n} kept={skip}", flush=True)


if __name__ == "__main__":
    main()