from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fgo_ocr.paths import MODELS

BLANK = 0
IMG_H = 64
MAX_W = 768

_sess = None
_sess_path = ""


def model_path() -> Path:
    env = (os.environ.get("FGO_OCR_MODEL") or "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for p in (
        MODELS / "exp_hires" / "ctc.onnx",
        MODELS / "rec.onnx",
    ):
        if p.is_file():
            return p
    return MODELS / "exp_hires" / "ctc.onnx"


def charset_path(onnx: Path | None = None) -> Path:
    onnx = onnx or model_path()
    local = onnx.parent / "charset.json"
    if local.is_file():
        return local
    return MODELS / "charset.json"


def _meta(onnx: Path | None = None) -> dict:
    p = charset_path(onnx)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def available() -> bool:
    return model_path().is_file() and charset_path().is_file()


def ctc_greedy(logits, chars: str) -> str:
    import numpy as np

    idx = np.asarray(logits).argmax(axis=-1)
    out: list[str] = []
    prev = None
    for i in idx.tolist():
        if i != BLANK and i != prev:
            j = i - 1
            if 0 <= j < len(chars):
                out.append(chars[j])
        prev = i
    return "".join(out)


def _to_rgb(image: Any):
    import numpy as np
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr[:, :, ::-1] if arr.dtype == np.uint8 else arr)
    return Image.fromarray(arr).convert("RGB")


def _trim_ink_width(im):
    import numpy as np
    from PIL import ImageFilter

    w, h = im.size
    if w < 32 or h < 8:
        return im
    gray = im.convert("L")
    g = np.asarray(gray, dtype=np.float32)
    blur = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(radius=max(2.0, h / 12.0))),
        dtype=np.float32,
    )
    e = np.abs(g - blur).mean(axis=0)
    k = max(5, w // 50)
    e = np.convolve(e, np.ones(k) / k, mode="same")
    peak = float(e.max())
    if peak < 1.2:
        return im
    p90 = float(np.percentile(e, 90))
    thr = max(peak * 0.25, p90 * 0.35, 1.5)
    active = e >= thr
    min_run = max(8, w // 20)
    gap = max(6, w // 30)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < w:
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < w and active[j]:
            j += 1
        if j - i >= min_run:
            if runs and i - runs[-1][1] <= gap:
                runs[-1] = (runs[-1][0], j)
            else:
                runs.append((i, j))
        i = j
    if not runs:
        return im
    x1, x2 = runs[0][0], runs[-1][1]
    if len(runs) >= 2:
        last_a, last_b = runs[-1]
        prev_b = runs[-2][1]
        last_w = last_b - last_a
        if (
            last_a - prev_b > gap * 2
            and last_w < int(w * 0.16)
            and last_a > int(w * 0.62)
        ):
            x2 = prev_b
    pad = max(6, w // 40)
    x1 = max(0, x1 - pad)
    x2 = min(w, x2 + pad)
    if x2 - x1 < 16 or (x2 - x1) >= int(w * 0.95):
        return im
    return im.crop((x1, 0, x2, h))


trim_banner = _trim_ink_width


def _session(path: Path):
    global _sess, _sess_path
    import onnxruntime as ort

    key = str(path)
    if _sess is None or _sess_path != key:
        _sess = ort.InferenceSession(key, providers=["CPUExecutionProvider"])
        _sess_path = key
    return _sess


def _input_hw(sess) -> tuple[int, int]:
    shape = list(sess.get_inputs()[0].shape)
    h = int(shape[2]) if len(shape) > 2 and isinstance(shape[2], int) else IMG_H
    w = int(shape[3]) if len(shape) > 3 and isinstance(shape[3], int) else MAX_W
    return max(8, h), max(8, w)


def _ctc_tensor(im, th: int, tw: int):
    import numpy as np
    from PIL import Image

    w = max(8, int(im.width * th / max(1, im.height)))
    w = min(tw, w)
    im = im.resize((w, th), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    canvas.paste(im, (0, 0))
    x = np.asarray(canvas).astype("float32") / 255.0
    return np.transpose(x, (2, 0, 1))[None]


def read(image: Any, *, trim: bool = True) -> str:
    onnx = model_path()
    if not onnx.is_file():
        raise FileNotFoundError(f"缺少 {onnx}")
    meta = _meta(onnx)
    chars = meta.get("chars") or ""
    if not chars:
        raise FileNotFoundError(f"缺少 charset {charset_path(onnx)}")
    im = _to_rgb(image)
    if trim:
        im = _trim_ink_width(im)
    sess = _session(onnx)
    name = sess.get_inputs()[0].name
    th, tw = _input_hw(sess)
    x = _ctc_tensor(im, th, tw)
    logits = sess.run(None, {name: x})[0]
    return ctc_greedy(logits[0], chars)


def main() -> None:
    print("model:", model_path(), "exists=" + str(model_path().is_file()))
    print("charset:", charset_path(), "exists=" + str(charset_path().is_file()))
    print("arch:", _meta().get("arch", "?"))
    if not available():
        print("還沒訓練。python -m fgo_ocr train_hires")


if __name__ == "__main__":
    main()