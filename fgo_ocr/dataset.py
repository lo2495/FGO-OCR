from __future__ import annotations

import os
import random
from pathlib import Path

from fgo_ocr.paths import CHARSET, DATA


def build_charset(texts: list[str]) -> str:
    seen: list[str] = []
    bag: set[str] = set()
    extra = CHARSET.read_text(encoding="utf-8") if CHARSET.is_file() else ""
    for src in (extra, *texts):
        for ch in src.replace("\n", ""):
            if ch not in bag:
                bag.add(ch)
                seen.append(ch)
    return "".join(seen)


def real_rows(root: Path | None = None) -> list[tuple[Path, str]]:
    root = root or (DATA / "real")
    if not root.is_dir():
        return []
    skip = {"holdout", "holdout_new"}
    rows: list[tuple[Path, str]] = []
    for p in root.rglob("*.png"):
        if any(part in skip for part in p.parts):
            continue
        txt = p.with_suffix(".txt")
        if not txt.is_file():
            continue
        text = txt.read_text(encoding="utf-8").strip().replace("\n", " ")
        if text:
            rows.append((p, text))
    return rows


def rows() -> list[tuple[Path, str]]:
    synth: list[tuple[Path, str]] = []
    for name in ("rec_gt.txt", "rec_gt_char.txt"):
        gt = DATA / name
        if not gt.is_file():
            continue
        for ln in gt.read_text(encoding="utf-8").splitlines():
            if "\t" not in ln:
                continue
            rel, text = ln.split("\t", 1)
            p = DATA / rel.strip()
            if p.is_file() and text.strip():
                synth.append((p, text.strip()))
    real = real_rows()
    if os.environ.get("FGO_OCR_SYNTH", "1") == "0":
        synth = []
    if not real and not synth:
        raise SystemExit("沒有樣本。先 synth 或在 data/real 寫上 txt 標籤")
    if real and synth:
        repeat = max(1, len(synth) // max(1, len(real)))
        real = real * repeat
    out = synth + real
    print(f"samples synth={len(synth)} real_eff={len(real)} total={len(out)}", flush=True)
    random.Random(7).shuffle(out)
    return out
