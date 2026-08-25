from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from fgo_ocr.infer import model_path, read
from fgo_ocr.paths import ATLAS_QUESTS, DATA
from fgo_ocr.synth import _fonts, render

N = 200
SEED = 99
OUT = DATA / "eval_new"


def _names() -> list[str]:
    if not ATLAS_QUESTS.is_file():
        raise SystemExit("沒有 atlas_quests.txt")
    out = []
    for ln in ATLAS_QUESTS.read_text(encoding="utf-8").splitlines():
        t = ln.strip()
        if 4 <= len(t) <= 36:
            out.append(t)
    return out


def main() -> None:
    fonts = _fonts()
    if not fonts:
        raise SystemExit("設 FGO_OCR_FONT")
    names = _names()
    rng = random.Random(SEED)
    sample = rng.sample(names, min(N, len(names)))
    OUT.mkdir(parents=True, exist_ok=True)
    print("using", model_path(), "n=", len(sample), flush=True)
    ok = n = 0
    char_ok = char_n = 0
    for i, gold in enumerate(sample):
        img = render(gold, rng.choice(fonts), rng)
        p = OUT / f"{i:04d}.png"
        img.save(p, quality=90)
        pred = read(Image.open(p))
        n += 1
        hit = pred == gold
        ok += int(hit)
        char_n += max(len(gold), 1)
        char_ok += sum(a == b for a, b in zip(pred, gold))
        if not hit:
            print(f"NG gold={gold}", flush=True)
            print(f"   pred={pred}", flush=True)
    print(
        f"exact={ok}/{n} {100 * ok / max(1, n):.1f}%  "
        f"char={char_ok}/{char_n} {100 * char_ok / max(1, char_n):.1f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
