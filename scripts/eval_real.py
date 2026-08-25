from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from fgo_ocr.infer import read
from fgo_ocr.paths import DATA, MODELS


def _norm(s: str) -> str:
    return "".join(s.split())


def main() -> None:
    onnx = MODELS / "rec.onnx"
    print(f"model={onnx} exists={onnx.is_file()}", flush=True)
    args = [Path(a) for a in sys.argv[1:] if a]
    files: list[Path] = []
    root = DATA / "real"
    for a in args:
        if a.is_dir():
            root = a
        elif a.is_file():
            files.append(a)
    if files:
        for p in files:
            print(p.name, "->", read(Image.open(p)), flush=True)
        return
    pngs = sorted(root.rglob("*.png"))
    n = hit = 0
    by: dict[str, list[int]] = {}
    for p in pngs:
        txt = p.with_suffix(".txt")
        gold = txt.read_text(encoding="utf-8").strip() if txt.is_file() else ""
        if not gold:
            continue
        pred = read(Image.open(p))
        n += 1
        ok = _norm(pred) == _norm(gold)
        hit += int(ok)
        kind = p.parent.name
        a, b = by.setdefault(kind, [0, 0])
        by[kind] = [a + int(ok), b + 1]
        mark = "OK" if ok else "NG"
        print(f"{mark} {kind}/{p.name}\n  gold={gold}\n  pred={pred}", flush=True)
    print(f"exact={hit}/{n}" + (f" {hit / n:.1%}" if n else ""), flush=True)
    for k, (a, b) in sorted(by.items()):
        print(f"  {k}: {a}/{b}" + (f" {a / b:.1%}" if b else ""), flush=True)


if __name__ == "__main__":
    main()