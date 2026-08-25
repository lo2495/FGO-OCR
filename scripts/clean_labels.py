from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real"
ATLAS = ROOT / "assets" / "atlas_quests.txt"

_BBOX = re.compile(r"<bbox>.*?</bbox>", re.I | re.S)
_POS = re.compile(r"<position[^>]*>", re.I)
_TAG = re.compile(r"</?(?:table|tr|td|th|br|div|span|p|html|body|h[1-6])[^>]*>", re.I)
_JUNK = (
    "このテキスト",
    "Markdown",
    "markdown",
    "この画像",
    "背景",
    "相当します",
    "書き直",
    "見出し",
    "bbox",
    "table",
    "Fate/Grand",
    "出力されます",
    "ピクセル",
    "ファイル",
)
_STOP = {
    "セイバー",
    "アーチャー",
    "ランサー",
    "ライダー",
    "キャスター",
    "アサシン",
    "バーサーカー",
    "エクストラ",
    " Extra",
    "段落",
    "ランスロット",
    "V",
    "I",
    "II",
    "III",
}
_CLASS = (
    "バーサーカー",
    "アサシン",
    "キャスター",
    "ライダー",
    "ランサー",
    "アーチャー",
    "セイバー",
)
_ROMAN = (
    ("VIII", "Ⅷ"),
    ("VII", "Ⅶ"),
    ("III", "Ⅲ"),
    ("II", "Ⅱ"),
    ("IV", "Ⅳ"),
    ("VI", "Ⅵ"),
    ("IX", "Ⅸ"),
    ("I", "Ⅰ"),
    ("8", "Ⅷ"),
    ("7", "Ⅶ"),
    ("6", "Ⅵ"),
    ("5", "Ⅴ"),
    ("4", "Ⅳ"),
    ("3", "Ⅲ"),
    ("2", "Ⅱ"),
    ("1", "Ⅰ"),
    ("八", "Ⅷ"),
    ("七", "Ⅶ"),
    ("六", "Ⅵ"),
    ("五", "Ⅴ"),
    ("四", "Ⅳ"),
)
_BR = str.maketrans({
    "[": "〔",
    "]": "〕",
    "［": "〔",
    "］": "〕",
    "【": "〔",
    "】": "〕",
    "｜": "",
    "|": "",
    "＞": "",
})
_GD = re.compile(
    r"冠位研[鑽鑚]戦\s*〔\s*(エクストラ\s*[IⅠ1Ⅱ2Ｉ]{0,3}|バーサーカー|アサシン|キャスター|"
    r"ライダー|ランサー|アーチャー|セイバー)\s*〕\s*([地水火風])?\s*"
    r"([IVXivxⅠⅡⅢⅣⅤⅥⅦⅧⅨ1-8七八六五四]*)",
)


def fold_roman(s: str) -> str:
    t = s.upper().replace("Ｉ", "I")
    for a, b in _ROMAN:
        t = t.replace(a, b)
    return t


def norm(s: str) -> str:
    t = (s or "").translate(_BR)
    t = t.replace("研鑚", "研鑽")
    t = fold_roman(t)
    t = re.sub(r"\s+", "", t)
    return t


def strip_noise(raw: str) -> str:
    t = _BBOX.sub(" ", raw or "")
    t = _POS.sub(" ", t)
    t = _TAG.sub(" ", t)
    t = t.replace("#", " ").replace("*", " ")
    return t


def _load_atlas() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ATLAS.is_file():
        return out
    for ln in ATLAS.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s:
            out[norm(s)] = s
    return out


def gd_title(text: str) -> str | None:
    t = norm(strip_noise(text))
    t = t.replace("〔", "〔").replace("〕", "〕")
    m = _GD.search(fold_roman(t.translate(_BR)))
    if not m:
        m = _GD.search(norm(text))
    if not m:
        return None
    inner, elem, num = m.group(1), m.group(2) or "", m.group(3) or ""
    inner = fold_roman(re.sub(r"\s+", "", inner))
    num = fold_roman(num)
    if inner.startswith("エクストラ"):
        if inner in ("エクストラ", "エクストラⅠⅠ"):
            if "Ⅱ" in inner or inner.endswith("2"):
                inner = "エクストラⅡ"
            elif re.search(r"Ⅰ|1", inner):
                inner = "エクストラⅠ"
            else:
                return None
        if inner not in ("エクストラⅠ", "エクストラⅡ"):
            if "Ⅱ" in inner or "II" in m.group(1).upper():
                inner = "エクストラⅡ"
            else:
                inner = "エクストラⅠ"
        if not elem:
            return None
        return f"冠位研鑽戦〔{inner}〕 {elem}{num}".strip()
    if not num:
        return None
    return f"冠位研鑽戦〔{inner}〕 {num}"


def pick_other(text: str, atlas: dict[str, str]) -> str:
    t = strip_noise(text)
    best = ""
    for chunk in re.split(r"[\n|]+", t):
        chunk = chunk.strip(" .-#")
        if len(chunk) < 2 or len(chunk) > 48:
            continue
        if chunk in _STOP or any(j in chunk for j in _JUNK):
            continue
        key = norm(chunk)
        if key in atlas:
            hit = atlas[key]
            if len(hit) > len(best):
                best = hit
    if best:
        return best
    for chunk in re.split(r"[\n|]+|(?:\s+-\s+)", t):
        chunk = re.sub(r"\s+", " ", chunk).strip(" .-#")
        if 3 <= len(chunk) <= 40 and chunk not in _STOP:
            if not any(j in chunk for j in _JUNK):
                if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", chunk):
                    return atlas.get(norm(chunk), chunk)
    return ""


def clean_one(raw: str, atlas: dict[str, str]) -> str:
    built = gd_title(raw)
    if built:
        hit = atlas.get(norm(built))
        if hit:
            return hit
        for k, v in atlas.items():
            if k.startswith(norm(built)[:12]) and abs(len(k) - len(norm(built))) <= 4:
                if norm(built) in k or k in norm(built):
                    return v
        return built
    return pick_other(raw, atlas)


def main() -> None:
    atlas = _load_atlas()
    print(f"atlas={len(atlas)} real={REAL}", flush=True)
    n = 0
    miss = 0
    for txt in sorted(REAL.rglob("*.txt")):
        raw = txt.read_text(encoding="utf-8")
        if not raw.strip():
            continue
        out = clean_one(raw, atlas)
        if out != raw.strip():
            txt.write_text((out + "\n") if out else "", encoding="utf-8")
            n += 1
            mark = "" if out else " MISS"
            if not out:
                miss += 1
            print(f"{txt.parent.name}/{txt.stem}  {out}{mark}", flush=True)
    print(f"changed={n} empty={miss}", flush=True)


if __name__ == "__main__":
    main()