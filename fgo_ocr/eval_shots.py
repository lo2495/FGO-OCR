from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

from fgo_ocr import infer


def _norm(s: str) -> str:
    return "".join(str(s).split())


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _gold(p: Path) -> str:
    for q in (p.with_suffix(".txt"), p.with_suffix(".gt.txt")):
        if q.is_file():
            return q.read_text(encoding="utf-8").strip().replace("\n", " ")
    return ""


def _pngs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.png"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/eval_holdout")
    ap.add_argument("--model", default="")
    ap.add_argument("--mask", type=int, default=0)
    ap.add_argument("--variants", default="raw")
    args, _ = ap.parse_known_args(argv)

    model = (args.model or os.environ.get("FGO_OCR_MODEL") or "").strip()
    if model:
        os.environ["FGO_OCR_MODEL"] = str(Path(model).resolve())

    onnx = infer.model_path()
    root = Path(args.root)
    if not root.is_absolute():
        root = Path.cwd() / root
    if not root.exists():
        raise SystemExit(f"找不到 {root}")

    n = hit = cer_n = cer_d = 0
    misses: list[str] = []
    for p in _pngs(root):
        gold = _gold(p)
        if not gold:
            continue
        im = Image.open(p)
        pred = infer.read(im)
        n += 1
        ok = _norm(pred) == _norm(gold)
        hit += int(ok)
        d = _lev(_norm(pred), _norm(gold))
        cer_n += d
        cer_d += max(1, len(_norm(gold)))
        mark = "OK" if ok else "NG"
        print(f"  {mark}  {p.name}  {im.size}  {gold!r} -> {pred!r}", flush=True)
        if not ok:
            misses.append(f"  {p.name:12}  gold={gold!r}  pred={pred!r}")

    exact = (hit / n * 100) if n else 0.0
    cer = (cer_n / cer_d * 100) if cer_d else 0.0
    print(
        f"eval_shots n={n} exact={exact:.1f}% cer={cer:.1f}%  "
        f"model={onnx} mask={args.mask} variants={args.variants}",
        flush=True,
    )
    out = root / "report.txt" if root.is_dir() else root.with_suffix(".report.txt")
    body = [
        f"n={n} exact={exact:.1f}% cer={cer:.1f}%",
        f"model={onnx}",
        "mismatches:",
        *misses,
        "",
    ]
    out.write_text("\n".join(body), encoding="utf-8")
    print(f"report {out}", flush=True)
    if misses:
        print("mismatches:", flush=True)
        print("\n".join(misses), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
