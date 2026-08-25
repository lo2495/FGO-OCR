from __future__ import annotations

import sys


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    rest = sys.argv[2:]
    if cmd in ("atlas", "fetch"):
        from fgo_ocr.atlas import main as atlas_main

        sys.argv = [sys.argv[0], *rest]
        atlas_main()
        return
    if cmd in ("synth", "generate"):
        from fgo_ocr.synth import main as synth_main

        synth_main()
        return
    if cmd in ("train", "train_parseq", "parseq"):
        from fgo_ocr.train_parseq import main as parseq_main

        parseq_main()
        return
    if cmd in ("infer", "check"):
        from fgo_ocr.infer import main as infer_main

        infer_main()
        return
    print("FGO-OCR")
    print("  python -m fgo_ocr atlas")
    print("  python -m fgo_ocr synth")
    print("  python -m fgo_ocr train")
    print("  python -m fgo_ocr infer")


if __name__ == "__main__":
    main()