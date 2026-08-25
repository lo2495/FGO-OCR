from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fgo_ocr.paths import MODELS


def charset_path() -> Path:
    return MODELS / "charset.json"


def model_path() -> Path:
    return MODELS / "parseq.onnx"


def _meta() -> dict:
    p = charset_path()
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def available() -> bool:
    return model_path().is_file() and charset_path().is_file()


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


def read(image: Any) -> str:
    if not available():
        raise FileNotFoundError(f"缺少 {model_path()}。先 python -m fgo_ocr train")
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image

    from fgo_ocr.parseq import IMG_H, IMG_W, greedy_text

    meta = _meta()
    chars = meta["chars"]
    im = _to_rgb(image)
    w = max(8, int(im.width * IMG_H / max(1, im.height)))
    w = min(IMG_W, w)
    im = im.resize((w, IMG_H), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (IMG_W, IMG_H), (20, 18, 16))
    canvas.paste(im, (0, 0))
    x = np.asarray(canvas).astype("float32") / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    sess = ort.InferenceSession(str(model_path()), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    t = torch.from_numpy(np.asarray(logits))
    if t.ndim == 3:
        t = t[0]
    return greedy_text(t, chars)


def main() -> None:
    print("model:", model_path(), "exists=" + str(model_path().is_file()))
    print("charset:", charset_path(), "exists=" + str(charset_path().is_file()))
    print("arch:", _meta().get("arch", "?"))
    if not available():
        print("還沒訓練。python -m fgo_ocr train")


if __name__ == "__main__":
    main()