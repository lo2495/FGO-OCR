from __future__ import annotations

import hashlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

AGENT = Path(os.environ.get("FGO_AGENT_ROOT", r"F:\MyOwnProject\FGO-Vision-Agent"))
OCR_ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("FGO_OCR_REAL", str(OCR_ROOT / "data" / "real")))
INTERVAL = float(os.environ.get("FGO_OCR_CAP_INTERVAL", "0.8"))
PAD = int(os.environ.get("FGO_OCR_CAP_PAD", "6"))
SET = os.environ.get("FGO_OCR_CAP_SET", "").strip()
TAB_Y0 = float(os.environ.get("FGO_OCR_TAB_Y0", "0.36"))
TAB_Y1 = float(os.environ.get("FGO_OCR_TAB_Y1", "0.66"))
TITLE_Y0 = float(os.environ.get("FGO_OCR_TITLE_Y0", "0.02"))
TITLE_Y1 = float(os.environ.get("FGO_OCR_TITLE_Y1", "0.60"))

sys.path.insert(0, str(AGENT))
os.chdir(AGENT)

from core.adb_client import AdbClient
from core.yolo_detector import YOLODetector
from utils.image_utils import crop_grand_duel_tab, crop_stage_title


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _hash(img: np.ndarray) -> str:
    small = cv2.resize(img, (64, 32), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()


def _band(img: np.ndarray, y0: float, y1: float) -> np.ndarray | None:
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    a = max(0, min(h - 1, int(h * y0)))
    b = max(a + 8, min(h, int(h * y1)))
    cut = img[a:b, 0:w]
    if cut.size == 0 or cut.shape[0] < 8:
        return None
    return cut


def _root() -> Path:
    return (OUT / SET) if SET else OUT


def _save(kind: str, img: np.ndarray, seen: set[str]) -> Path | None:
    if img is None or img.size == 0:
        return None
    h = _hash(img)
    if h in seen:
        return None
    seen.add(h)
    folder = _root() / kind
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_stamp()}.png"
    cv2.imwrite(str(path), img)
    path.with_suffix(".txt").write_text("", encoding="utf-8")
    return path


def _items(detections: dict, key: str) -> list[dict]:
    raw = detections.get(f"{key}_list") or detections.get(key)
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [raw]
    return []


def capture_frame(frame: np.ndarray, detections: dict, seen: set[str]) -> list[str]:
    saved: list[str] = []
    for det in _items(detections, "btn_GrandDuel_Selection"):
        box = det.get("box")
        if not box:
            continue
        _focus, crop = crop_grand_duel_tab(frame, box, scale=1.0, clahe=False)
        crop = _band(crop, TAB_Y0, TAB_Y1)
        p = _save("tab", crop, seen)
        if p:
            saved.append(f"tab {p.name}")
    for det in _items(detections, "btn_GameStage"):
        box = det.get("box")
        if not box:
            continue
        crop = crop_stage_title(frame, box)
        p = _save("stage", crop, seen)
        if p:
            saved.append(f"stage {p.name}")
    title = detections.get("text_headtitle")
    if isinstance(title, dict) and title.get("box"):
        x1, y1, x2, y2 = title["box"]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
        x2, y2 = min(w, x2 + PAD), min(h, y2 + PAD)
        crop = _band(frame[y1:y2, x1:x2], TITLE_Y0, TITLE_Y1)
        p = _save("title", crop, seen)
        if p:
            saved.append(f"title {p.name}")
    return saved


def main() -> None:
    _root().mkdir(parents=True, exist_ok=True)
    print(f"agent={AGENT}", flush=True)
    print(f"out={_root()}", flush=True)
    print(f"tab_band={TAB_Y0:.2f}-{TAB_Y1:.2f} title_band={TITLE_Y0:.2f}-{TITLE_Y1:.2f}", flush=True)
    print("在遊戲裡切冠位分頁 / 關卡列表。Ctrl+C 結束。", flush=True)
    client = AdbClient()
    yolo = YOLODetector(conf_threshold=0.35)
    seen: set[str] = set()
    n = 0
    try:
        while True:
            frame = client.screencap()
            if frame is None:
                print("截圖失敗", flush=True)
                time.sleep(INTERVAL)
                continue
            dets = yolo.detect_ui(frame)
            hits = capture_frame(frame, dets, seen)
            keys = sorted(k for k in dets if not k.endswith("_list"))
            if hits:
                n += len(hits)
                print(f"+{len(hits)} total={n}  {', '.join(hits)}", flush=True)
            else:
                print(f"… yolo={keys[:8]}", flush=True)
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print(f"\nstop total={n} -> {_root()}", flush=True)
        print("請編輯 txt 寫上日文標籤（空檔不要拿去訓練）", flush=True)


if __name__ == "__main__":
    main()