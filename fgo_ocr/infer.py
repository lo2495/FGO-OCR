from __future__ import annotations

from pathlib import Path
from typing import Any

from fgo_ocr.paths import MODELS


def model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "rec.onnx"


def available() -> bool:
    return model_path().is_file()


def read(image: Any) -> str:
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 {path}。先合成資料、微調 rec，再把 ONNX 放到 models/rec.onnx"
        )
    import numpy as np
    import onnxruntime as ort

    arr = image
    if not isinstance(arr, np.ndarray):
        from PIL import Image

        arr = np.array(Image.fromarray(np.asarray(image)).convert("RGB"))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    name, shape = inp.name, inp.shape
    h, w = int(shape[2] or 32), int(shape[3] or 320)
    from PIL import Image

    im = Image.fromarray(arr).convert("RGB").resize((w, h))
    x = np.asarray(im).astype("float32") / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    out = sess.run(None, {name: x})[0]
    return str(out)


def main() -> None:
    print("model:", model_path(), "exists=" + str(available()))
    if not available():
        print("還沒有 models/rec.onnx")


if __name__ == "__main__":
    main()
