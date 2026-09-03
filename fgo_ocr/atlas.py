from __future__ import annotations

import json
import ssl
import urllib.request

from fgo_ocr.paths import ASSETS, ATLAS_NAMES, ATLAS_QUESTS

_UA = {"User-Agent": "FGO-OCR/1.0"}
_CTX = ssl._create_unverified_context()
_QUEST_URLS = (
    "https://api.atlasacademy.io/export/JP/basic_quest.json",
    "https://api.atlasacademy.io/export/JP/nice_war.json",
)
_NAME_URLS = ("https://api.atlasacademy.io/export/JP/basic_servant.json",)


def _get(url: str):
    print(f"atlas fetch {url}", flush=True)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=180, context=_CTX) as r:
        return json.loads(r.read().decode("utf-8"))


def _walk(obj, out: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("name", "longName", "overwriteName") and isinstance(v, str):
                t = " ".join(v.split()).strip()
                if 2 <= len(t) <= 48 and t not in ("0", "-", "？", "?"):
                    out.add(t)
            else:
                _walk(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _walk(x, out)


def _dump(urls: tuple[str, ...], dest) -> list[str]:
    bag: set[str] = set()
    if dest.is_file():
        for ln in dest.read_text(encoding="utf-8").splitlines():
            t = ln.strip()
            if t:
                bag.add(t)
    for url in urls:
        try:
            _walk(_get(url), bag)
        except Exception as e:
            print(f"atlas skip {url}: {e}", flush=True)
    lines = sorted(bag)
    if not lines:
        print(f"atlas empty, keep {dest}", flush=True)
        return lines
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} -> {dest}", flush=True)
    return lines


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    _dump(_QUEST_URLS, ATLAS_QUESTS)
    _dump(_NAME_URLS, ATLAS_NAMES)


if __name__ == "__main__":
    main()