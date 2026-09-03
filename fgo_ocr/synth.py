from __future__ import annotations

import io
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from fgo_ocr.paths import ATLAS_NAMES, ATLAS_QUESTS, DATA, LABELS

STYLES = (
    "banner_white",
    "banner_ink",
    "banner_gold",
    "dark_gold",
    "dark_white",
    "dark_cyan",
    "event_red",
    "event_blue",
    "event_purple",
    "parchment",
    "overexpose",
    "night_map",
    "green_free",
    "steel",
    "pink_event",
    "scanline",
)


def _fonts() -> list[str]:
    env = [p for p in os.environ.get("FGO_OCR_FONT", "").split(os.pathsep) if p and Path(p).is_file()]
    win = [
        r"C:\Windows\Fonts\YuGothB.ttc",
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\YuGothR.ttc",
        r"C:\Windows\Fonts\YuGothL.ttc",
        r"C:\Windows\Fonts\meiryob.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\yumin.ttf",
        r"C:\Windows\Fonts\YuMincho.ttf",
        r"C:\Windows\Fonts\msmincho.ttc",
        r"C:\Windows\Fonts\BIZ-UDGothicR.ttc",
        r"C:\Windows\Fonts\BIZ-UDGothicB.ttc",
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NotoSansCJKjp-Regular.otf",
    ]
    out, seen = [], set()
    for p in env + win:
        k = Path(p).name.lower()
        if k in seen or not Path(p).is_file():
            continue
        seen.add(k)
        out.append(p)
    return out


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size, index=0)
    except Exception:
        return ImageFont.truetype(path, size)


def load_labels(extra: Path | None = None) -> list[str]:
    if not ATLAS_QUESTS.is_file():
        try:
            from fgo_ocr.atlas import main as atlas_main

            atlas_main()
        except Exception as e:
            print(f"atlas fetch skip: {e}", flush=True)
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


def _style(rng: random.Random) -> str:
    if rng.random() < 0.55:
        return "random"
    r = rng.random()
    if r < 0.18:
        return "banner_white"
    if r < 0.28:
        return "banner_ink"
    if r < 0.36:
        return "banner_gold"
    return rng.choice(STYLES)


def _noise(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    n = np.random.randint(lo, hi, size=arr.shape, dtype=np.int16)
    return np.clip(arr.astype(np.int16) + n, 0, 255).astype(np.uint8)


def _fill(h: int, w: int, rgb: tuple[int, int, int]) -> np.ndarray:
    arr = np.empty((h, w, 3), dtype=np.int16)
    arr[..., 0], arr[..., 1], arr[..., 2] = rgb
    return arr


def _grad(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = arr.shape[:2]
    xs = (np.linspace(-1, 1, w) * rng.uniform(6, 22)).astype(np.int16)
    ys = (np.linspace(-1, 1, h) * rng.uniform(0, 14)).astype(np.int16)
    arr = arr + xs.reshape(1, w, 1) + ys.reshape(h, 1, 1)
    return arr


def _stripes(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    step = rng.randint(6, 14)
    mag = rng.randint(4, 16)
    arr = arr + ((((xx + yy) // step) % 2) * mag).reshape(h, w, 1)
    return arr


def _sparkles(arr: np.ndarray, rng: random.Random, n: int | None = None) -> np.ndarray:
    h, w = arr.shape[:2]
    n = rng.randint(8, 48) if n is None else n
    for _ in range(n):
        x = rng.randint(0, w - 1)
        y = rng.randint(0, h - 1)
        rad = rng.randint(1, 2)
        y0, y1 = max(0, y - rad), min(h, y + rad + 1)
        x0, x1 = max(0, x - rad), min(w, x + rad + 1)
        arr[y0:y1, x0:x1] += rng.randint(16, 70)
    return arr


def _vignette(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = arr.shape[:2]
    yy, xx = np.ogrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    d = ((yy - cy) / max(1, h)) ** 2 + ((xx - cx) / max(1, w)) ** 2
    arr = arr - (d * rng.uniform(8, 28)).astype(np.int16)[..., None]
    return arr


def _to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _bg(style: str, w: int, h: int, rng: random.Random) -> Image.Image:
    if style == "random":
        base = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        arr = _fill(h, w, base)
        if rng.random() < 0.75:
            arr = _grad(arr, rng)
        if rng.random() < 0.45:
            arr = _stripes(arr, rng)
        if rng.random() < 0.4:
            arr = _sparkles(arr, rng)
        if rng.random() < 0.3:
            arr = _vignette(arr, rng)
        if rng.random() < 0.2:
            arr[:: rng.randint(2, 4)] += rng.randint(8, 28)
        return _to_img(_noise(arr, -14, 15))
    if style.startswith("banner") or style == "overexpose":
        base = (
            rng.randint(176, 224),
            rng.randint(180, 228),
            rng.randint(196, 240),
        )
        if style == "overexpose":
            base = tuple(min(255, c + 28) for c in base)
        arr = _sparkles(_stripes(_grad(_fill(h, w, base), rng), rng), rng)
        return _to_img(arr)
    if style == "parchment":
        arr = _grad(_fill(h, w, (rng.randint(196, 220), rng.randint(176, 198), rng.randint(140, 168))), rng)
        return _to_img(_noise(arr, -10, 11))
    if style == "night_map":
        arr = _vignette(_grad(_fill(h, w, (rng.randint(12, 32), rng.randint(16, 40), rng.randint(28, 58))), rng), rng)
        return _to_img(_sparkles(arr, rng, rng.randint(4, 16)))
    if style == "event_red":
        arr = _grad(_fill(h, w, (rng.randint(70, 120), rng.randint(10, 36), rng.randint(16, 40))), rng)
        return _to_img(_stripes(arr, rng))
    if style == "event_blue":
        arr = _grad(_fill(h, w, (rng.randint(16, 48), rng.randint(28, 70), rng.randint(90, 150))), rng)
        return _to_img(_stripes(arr, rng))
    if style == "event_purple":
        arr = _grad(_fill(h, w, (rng.randint(48, 90), rng.randint(16, 48), rng.randint(80, 130))), rng)
        return _to_img(_sparkles(arr, rng, 12))
    if style == "green_free":
        arr = _grad(_fill(h, w, (rng.randint(20, 50), rng.randint(70, 120), rng.randint(40, 80))), rng)
        return _to_img(arr)
    if style == "steel":
        arr = _stripes(_fill(h, w, (rng.randint(70, 110), rng.randint(76, 116), rng.randint(88, 130))), rng)
        return _to_img(_noise(arr, -8, 9))
    if style == "pink_event":
        arr = _grad(_fill(h, w, (rng.randint(180, 220), rng.randint(90, 140), rng.randint(130, 180))), rng)
        return _to_img(arr)
    if style == "scanline":
        arr = _fill(h, w, (rng.randint(20, 40), rng.randint(22, 44), rng.randint(28, 50)))
        arr[::2] += 18
        return _to_img(arr)
    if style == "dark_cyan":
        arr = _fill(h, w, (rng.randint(8, 24), rng.randint(28, 52), rng.randint(40, 70)))
        return _to_img(_noise(arr, -8, 9))
    base = rng.randint(14, 44)
    arr = _fill(h, w, (base, max(8, base - 4), max(8, base - 12)))
    return _to_img(_noise(arr, -8, 9))


def _fill_color(style: str, rng: random.Random, bg: Image.Image | None = None) -> tuple[int, int, int]:
    if style == "random" or (bg is not None and rng.random() < 0.7):
        mean = float(np.asarray(bg).mean()) if bg is not None else 128.0
        kind = rng.randrange(4)
        if kind == 0:
            v = rng.randint(210, 255) if mean < 140 else rng.randint(0, 48)
            return (v, min(255, v + rng.randint(-12, 12)), min(255, v + rng.randint(-18, 8)))
        if kind == 1:
            return (rng.randint(210, 255), rng.randint(160, 230), rng.randint(40, 120))
        if kind == 2:
            return (rng.randint(120, 210), rng.randint(210, 255), rng.randint(220, 255))
        v = rng.randint(0, 40) if mean > 140 else rng.randint(230, 255)
        return (v, v, v)
    if style in ("banner_white", "overexpose", "dark_white", "night_map", "event_red", "event_blue", "event_purple"):
        v = rng.randint(236, 255)
        return (v, min(255, v + rng.randint(-4, 4)), min(255, v + rng.randint(-8, 2)))
    if style == "banner_ink":
        v = rng.randint(48, 96)
        return (v, v + rng.randint(0, 10), min(255, v + rng.randint(8, 24)))
    if style in ("banner_gold", "dark_gold", "parchment"):
        return (rng.randint(210, 255), rng.randint(168, 220), rng.randint(60, 130))
    if style == "dark_cyan":
        return (rng.randint(140, 200), rng.randint(220, 255), rng.randint(230, 255))
    if style == "green_free":
        return (rng.randint(220, 255), rng.randint(230, 255), rng.randint(180, 230))
    if style == "pink_event":
        return (255, rng.randint(230, 255), rng.randint(236, 255))
    v = rng.randint(220, 255)
    return (v, v, v)


def _shade(style: str, rng: random.Random) -> tuple[int, int, int]:
    if style.startswith("banner") or style == "overexpose":
        v = rng.randint(64, 120)
        return (v, v + 4, v + 16)
    return (rng.randint(8, 28), rng.randint(8, 24), rng.randint(8, 32))


def _effects(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.18:
        shear = rng.uniform(-0.08, 0.08)
        img = img.transform(
            img.size,
            Image.AFFINE,
            (1, shear, 0, 0, 1, 0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=img.getpixel((0, 0)),
        )
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 1.25)))
    if rng.random() < 0.12:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=rng.randint(80, 160)))
    if rng.random() < 0.35:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.45))
    if rng.random() < 0.25:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.82, 1.22))
    if rng.random() < 0.2:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.55, 1.25))
    if rng.random() < 0.12:
        img = ImageOps.autocontrast(img, cutoff=rng.randint(0, 2))
    if rng.random() < 0.38:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(32, 88))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    if rng.random() < 0.22:
        s = rng.uniform(0.72, 0.92)
        small = img.resize((max(32, int(img.width * s)), max(20, int(img.height * s))), Image.Resampling.BILINEAR)
        img = small.resize(img.size, Image.Resampling.NEAREST if rng.random() < 0.4 else Image.Resampling.BILINEAR)
    scale = rng.uniform(0.82, 1.18)
    nw, nh = max(32, int(img.width * scale)), max(24, int(img.height * scale))
    return img.resize((nw, nh), Image.Resampling.BILINEAR).convert("RGB")


_HIRA = "".join(chr(c) for c in range(0x3041, 0x3097))
_KATA = "".join(chr(c) for c in range(0x30A1, 0x30FB))
_PUNCT = "〔〕【】［］（）『』「」/／・!！?？~～、。 "
_ROMAN = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ"
_ASCII_ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")
_BRACKETS = (("〔", "〕"), ("【", "】"), ("（", "）"), ("『", "』"), ("「", "」"))
_ELEM = "地水火風天空"


def _rand_kana(rng: random.Random, n: int | None = None, kata: bool | None = None) -> str:
    use_kata = rng.random() < 0.5 if kata is None else kata
    pool = _KATA if use_kata else _HIRA
    return "".join(rng.choice(pool) for _ in range(n or rng.randint(4, 12)))


def _rand_roman(rng: random.Random) -> str:
    if rng.random() < 0.65:
        return rng.choice(_ROMAN)
    return rng.choice(_ASCII_ROMAN)


def _clip(s: str, rng: random.Random) -> str:
    if len(s) < 4:
        return s
    a = rng.randint(0, max(0, len(s) // 3))
    b = rng.randint(max(a + 2, len(s) * 2 // 3), len(s))
    return s[a:b]


def _punct_wrap(rng: random.Random, text: str) -> str:
    kind = rng.randrange(3)
    if kind == 0:
        a, b = rng.choice(_BRACKETS)
        return f"{a}{text}{b}"
    if kind == 1:
        i = rng.randint(0, len(text))
        return text[:i] + rng.choice(_PUNCT) + text[i:]
    return text + rng.choice(_PUNCT)


def _piece(rng: random.Random, labels: list[str]) -> str:
    k = rng.randrange(5)
    if k == 0:
        return rng.choice(labels)
    if k == 1:
        return _rand_kana(rng, kata=True)
    if k == 2:
        return _rand_kana(rng, kata=False)
    if k == 3:
        return _rand_roman(rng)
    return rng.choice(_PUNCT)


def _elem_roman(rng: random.Random) -> str:
    return rng.choice(_ELEM) + _rand_roman(rng)


def _pick_text(rng: random.Random, labels: list[str]) -> str:
    mix = (os.environ.get("FGO_OCR_MIX") or "full").strip().lower()
    if mix == "full":
        r = rng.random()
        if r < 0.40:
            return rng.choice(labels)
        if r < 0.50:
            return _rand_kana(rng, kata=True)
        if r < 0.60:
            return _rand_kana(rng, kata=False)
        if r < 0.70:
            t = rng.choice(labels)
            return _punct_wrap(rng, t) if rng.random() < 0.6 else t
        if r < 0.80:
            return _elem_roman(rng)
        if r < 0.90:
            return f"{rng.choice(labels)} {_elem_roman(rng)}"
        if r < 0.95:
            return _rand_roman(rng)
        return _punct_wrap(rng, _rand_kana(rng))
    n = rng.choice((1, 1, 2, 2, 3))
    parts = [_piece(rng, labels) for _ in range(n)]
    if n == 1:
        text = parts[0]
        if rng.random() < 0.35:
            text = _punct_wrap(rng, text)
        return text
    out = parts[0]
    for p in parts[1:]:
        glue = rng.choice(("", " ", rng.choice(_PUNCT)))
        out = f"{out}{glue}{p}"
    if rng.random() < 0.2:
        out = _clip(out, rng)
    return out


def render(text: str, font_path: str, rng: random.Random) -> Image.Image:
    style = _style(rng)
    size = rng.randint(22, 44)
    font = _load_font(font_path, size)
    dummy = Image.new("RGB", (8, 8))
    dr = ImageDraw.Draw(dummy)
    box = dr.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad_x, pad_y = rng.randint(8, 32), rng.randint(5, 18)
    w, h = max(64, tw + pad_x * 2), max(32, th + pad_y * 2)
    img = _bg(style, w, h, rng)
    draw = ImageDraw.Draw(img)
    x = pad_x - box[0] + rng.randint(-2, 2)
    y = pad_y - box[1] + rng.randint(-2, 2)
    fill = _fill_color(style, rng, img)
    shade = _shade(style, rng)
    if rng.random() < 0.75:
        for dx, dy in ((1, 1), (0, 1), (1, 0), (-1, 1)):
            draw.text((x + dx, y + dy), text, font=font, fill=shade)
    draw.text((x, y), text, font=font, fill=fill)
    if rng.random() < 0.28:
        overlay = np.array(img).astype(np.int16)
        overlay = _sparkles(overlay, rng, rng.randint(6, 24))
        img = _to_img(overlay)
    return _effects(img, rng)


def _job(item: tuple) -> str:
    i, text, font, seed, train_dir = item
    rng = random.Random(seed)
    np.random.seed(seed % (2**32 - 1))
    img = render(text, font, rng)
    name = f"{i:06d}.jpg"
    img.save(Path(train_dir) / name, quality=rng.randint(62, 95))
    return f"train/{name}\t{text}"


def generate(
    n: int | None = None,
    out: Path | None = None,
    extra_labels: Path | None = None,
    seed: int | None = None,
) -> Path:
    n = int(os.environ.get("FGO_OCR_N", "160000") if n is None else n)
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
    workers = int(os.environ.get("FGO_OCR_SYNTH_WORKERS", str(max(1, (os.cpu_count() or 8) - 1))))
    print(
        f"synth start n={n} labels={len(labels)} fonts={len(fonts)} "
        f"workers={workers} -> {train_dir}",
        flush=True,
    )
    jobs = [
        (i, _pick_text(rng, labels), rng.choice(fonts), seed + i * 9973, str(train_dir))
        for i in range(n)
    ]
    gt: list[str] = []
    step = max(1, n // 20)
    chunk = max(16, n // (workers * 40))
    Pool = ThreadPoolExecutor if os.name == "nt" else ProcessPoolExecutor
    with Pool(max_workers=workers) as ex:
        for i, line in enumerate(ex.map(_job, jobs, chunksize=chunk), 1):
            gt.append(line)
            if i % step == 0 or i == n:
                print(f"  {i}/{n}", flush=True)
    (out / "rec_gt.txt").write_text("\n".join(gt) + "\n", encoding="utf-8")
    print(f"wrote {n} images -> {out} fonts={len(fonts)} labels={len(labels)}", flush=True)
    return out


def main() -> None:
    extra = os.environ.get("FGO_OCR_LABELS", "").strip()
    generate(extra_labels=Path(extra) if extra else None)


if __name__ == "__main__":
    main()
