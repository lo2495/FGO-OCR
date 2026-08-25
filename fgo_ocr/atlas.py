from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

from fgo_ocr.paths import ASSETS, CHARSET, DATA

BASE = "https://api.atlasacademy.io/export/JP"
SOURCES = (
    ("nice_war.json", "atlas_nice_war.json"),
    ("basic_servant.json", "atlas_basic_servant.json"),
    ("basic_equip.json", "atlas_basic_equip.json"),
    ("nice_mystic_code.json", "atlas_nice_mc.json"),
    ("nice_item.json", "atlas_nice_item.json"),
    ("nice_command_code.json", "atlas_nice_cc.json"),
)
LORE_SOURCES = (
    ("nice_servant.json", "atlas_nice_servant.json"),
    ("nice_equip.json", "atlas_nice_equip.json"),
)
QUESTS_OUT = ASSETS / "atlas_quests.txt"
NAMES_OUT = ASSETS / "atlas_names.txt"
CHARSET_ATLAS = ASSETS / "charset_atlas.txt"
MAX_LINE = 48
SKIP = {"？", "?", "-", "None", "null"}


def _ssl(verify: bool):
    if verify:
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()
    return ssl._create_unverified_context()


def _get(url: str, cache: Path, force: bool = False):
    if cache.is_file() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        print(f"atlas cache {cache.name}")
        return data
    req = urllib.request.Request(url, headers={"User-Agent": "FGO-OCR/0.1"})
    last: Exception | None = None
    for verify in (True, False):
        try:
            with urllib.request.urlopen(req, timeout=300, context=_ssl(verify)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            print(f"atlas downloaded {url.split('/')[-1]} -> {cache.name}")
            return data
        except Exception as e:
            last = e
    raise RuntimeError(f"atlas download failed {url}: {last}")


def _walk(obj, names: set[str], chars: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _walk(v, names, chars)
        return
    if isinstance(obj, list):
        for x in obj:
            _walk(x, names, chars)
        return
    if not isinstance(obj, str):
        return
    t = obj.strip()
    if not t or t in SKIP:
        return
    chars.update(t)
    if len(t) <= MAX_LINE:
        names.add(t)


def _dump_lines(path: Path, items: list[str], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + "\n", encoding="utf-8")
    print(f"{label} {len(items)} -> {path}")


def refresh(force: bool = False, lore: bool = True) -> None:
    names: set[str] = set()
    chars: set[str] = set()
    packs = list(SOURCES)
    if lore:
        packs.extend(LORE_SOURCES)
    for file, cache_name in packs:
        data = _get(f"{BASE}/{file}", DATA / cache_name, force=force)
        _walk(data, names, chars)
    short = sorted(n for n in names if n)
    _dump_lines(QUESTS_OUT, short, "atlas lines")
    _dump_lines(NAMES_OUT, short, "atlas names")
    extra = "".join(sorted(chars))
    CHARSET_ATLAS.write_text(extra + "\n", encoding="utf-8")
    print(f"atlas charset unique={len(chars)} -> {CHARSET_ATLAS}")
    seed = CHARSET.read_text(encoding="utf-8") if CHARSET.is_file() else ""
    merged = []
    seen = set()
    for ch in seed.replace("\n", "") + extra:
        if ch not in seen:
            seen.add(ch)
            merged.append(ch)
    CHARSET.write_text("".join(merged) + "\n", encoding="utf-8")
    print(f"merged charset {len(merged)} -> {CHARSET}")


def main() -> None:
    import sys

    force = "--force" in sys.argv
    lore = "--no-lore" not in sys.argv
    refresh(force=force, lore=lore)


if __name__ == "__main__":
    main()