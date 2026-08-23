from __future__ import annotations

import sys


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd in ("synth", "generate"):
        from fgo_ocr.synth import main as synth_main

        synth_main()
        return
    if cmd in ("infer", "check"):
        from fgo_ocr.infer import main as infer_main

        infer_main()
        return
    print("FGO-OCR  獨立專案（CPU 封閉詞表）")
    print("  python -m fgo_ocr synth")
    print("  python -m fgo_ocr infer")


if __name__ == "__main__":
    main()
