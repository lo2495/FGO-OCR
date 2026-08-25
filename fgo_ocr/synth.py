from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from fgo_ocr.paths import ATLAS_NAMES, ATLAS_QUESTS, DATA, LABELS


def _fonts() -> list[str]:
    env = os.environ.get("FGO_OCR_FONT", "")
    found = [p for p in env.split(os.pathsep) if p and Path(p).is_file()]
    win = [
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\YuGothR.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\yumin.ttf",
        r"C:\Windows\Fonts\msmincho.ttc",
    ]
    return [p for p in found + win if Path(p).is_file()]


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size, index=0)
    except Exception:
        return ImageFont.truetype(path, size)


def load_labels(extra: Path | None = None) -> list[str]:
    paths = [LABELS, ATLAS_QUESTS, ATLAS_NAMES]
    if extra is not None and extra.is_file():
        paths.append(extra)
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        if not p.is_file():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            t = ln.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _bg(w: int, h: int, rng: random.Random) -> Image.Image:
    base = rng.randint(18, 42)
    rgb = np.array([base, max(8, base - 4), max(8, base - 12)], dtype=np.int16)
    noise = np.random.randint(-8, 9, size=(h, w, 3), dtype=np.int16)
    arr = np.clip(rgb + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _gold(rng: random.Random) -> tuple[int, int, int]:
    return (
        rng.randint(210, 255),
        rng.randint(170, 220),
        rng.randint(70, 130),
    )


def render(text: str, font_path: str, rng: random.Random) -> Image.Image:
    size = rng.randint(22, 40)
    font = _load_font(font_path, size)
    dummy = Image.new("RGB", (8, 8))
    dr = ImageDraw.Draw(dummy)
    box = dr.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad_x, pad_y = rng.randint(10, 28), rng.randint(6, 16)
    w, h = max(64, tw + pad_x * 2), max(32, th + pad_y * 2)
    img = _bg(w, h, rng)
    draw = ImageDraw.Draw(img)
    x = pad_x - box[0]
    y = pad_y - box[1]
    draw.text((x + 1, y + 1), text, font=font, fill=(20, 12, 8))
    draw.text((x, y), text, font=font, fill=_gold(rng))
    if rng.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.1)))
    if rng.random() < 0.35:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.35))
    if rng.random() < 0.3:
        arr = np.array(img).astype(np.int16)
        arr += rng.randint(-12, 12)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    scale = rng.uniform(0.85, 1.15)
    nw, nh = max(32, int(img.width * scale)), max(24, int(img.height * scale))
    return img.resize((nw, nh), Image.Resampling.BILINEAR).convert("RGB")


def generate(
    n: int | None = None,
    out: Path | None = None,
    extra_labels: Path | None = None,
    seed: int | None = None,
) -> Path:
    n = int(os.environ.get("FGO_OCR_N", "4000") if n is None else n)
    seed = int(os.environ.get("FGO_OCR_SEED", "7") if seed is None else seed)
    out = Path(os.environ.get("FGO_OCR_OUT", str(out or DATA)))
    rng = random.Random(seed)
    np.random.seed(seed)
    fonts = _fonts()
    if not fonts:
        raise SystemExit("找不到日文字型。設 FGO_OCR_FONT=C:\\Windows\\Fonts\\YuGothM.ttc")
    labels = load_labels(extra_labels)
    if not labels:
        raise SystemExit("assets/labels.txt 是空的")
    train_dir = out / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    print(f"synth start n={n} labels={len(labels)} fonts={len(fonts)} -> {train_dir}", flush=True)
    gt = []
    step = max(1, n // 20)
    for i in range(n):
        text = rng.choice(labels)
        img = render(text, rng.choice(fonts), rng)
        name = f"{i:06d}.jpg"
        img.save(train_dir / name, quality=rng.randint(70, 95))
        gt.append(f"train/{name}\t{text}")
        if (i + 1) % step == 0 or i == 0:
            print(f"  {i + 1}/{n}", flush=True)
    (out / "rec_gt.txt").write_text("\n".join(gt) + "\n", encoding="utf-8")
    print(f"wrote {n} images -> {out} fonts={len(fonts)} labels={len(labels)}", flush=True)
    return out


def main() -> None:
    extra = os.environ.get("FGO_OCR_LABELS", "").strip()
    generate(extra_labels=Path(extra) if extra else None)


if __name__ == "__main__":
    main()