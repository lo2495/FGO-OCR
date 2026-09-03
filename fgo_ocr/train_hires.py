from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from fgo_ocr.paths import CHARSET, DATA, MODELS, ROOT
from fgo_ocr.synth import load_labels

IMG_H = 64
MAX_W = 768
BLANK = 0
OUT = MODELS / "exp_hires"


def build_charset(texts: list[str]) -> str:
    seen: list[str] = []
    bag: set[str] = set()
    extra = CHARSET.read_text(encoding="utf-8") if CHARSET.is_file() else ""
    for src in (extra, *texts):
        for ch in src.replace("\n", ""):
            if ch not in bag:
                bag.add(ch)
                seen.append(ch)
    return "".join(seen)


def encode(text: str, table: dict[str, int]) -> list[int]:
    return [table[c] for c in text if c in table]


def ctc_greedy(logits: np.ndarray, chars: str) -> str:
    idx = logits.argmax(axis=-1)
    out: list[str] = []
    prev = None
    for i in idx.tolist():
        if i != BLANK and i != prev:
            j = i - 1
            if 0 <= j < len(chars):
                out.append(chars[j])
        prev = i
    return "".join(out)


class RecCRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)
        f = f.squeeze(2).permute(0, 2, 1)
        y, _ = self.rnn(f)
        return self.fc(y)


class RecSet(Dataset):
    def __init__(self, rows: list[tuple[Path, str]], table: dict[str, int]):
        self.rows = rows
        self.table = table

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        path, text = self.rows[i]
        img = Image.open(path).convert("RGB")
        h, w = IMG_H, max(8, int(img.width * IMG_H / max(1, img.height)))
        w = min(MAX_W, w)
        img = img.resize((w, h), Image.Resampling.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        y = torch.tensor(encode(text, self.table), dtype=torch.long)
        if y.numel() == 0:
            y = torch.tensor([1], dtype=torch.long)
        return x, y


def _collate(batch):
    xs, ys = zip(*batch)
    max_w = max(x.shape[2] for x in xs)
    canvas = torch.zeros(len(xs), 3, IMG_H, max_w)
    for i, x in enumerate(xs):
        canvas[i, :, :, : x.shape[2]] = x
    targets = pad_sequence(list(ys), batch_first=True, padding_value=BLANK)
    lengths = torch.tensor([y.numel() for y in ys], dtype=torch.long)
    return canvas, targets, lengths


def _skip_part(p: Path) -> bool:
    skip = {"holdout", "eval_holdout", "eval", "inbox"}
    return any(part in skip for part in p.parts)


def _rows() -> list[tuple[Path, str]]:
    gt = DATA / "rec_gt.txt"
    synth: list[tuple[Path, str]] = []
    if gt.is_file():
        for ln in gt.read_text(encoding="utf-8").splitlines():
            if "\t" not in ln:
                continue
            rel, text = ln.split("\t", 1)
            p = DATA / rel.strip()
            if p.is_file() and text.strip():
                synth.append((p, text.strip()))
    real: list[tuple[Path, str]] = []
    root = DATA / "real"
    if not root.is_dir():
        alt = ROOT / "data" / "real"
        if alt.is_dir():
            root = alt
    if root.is_dir():
        for p in root.rglob("*.png"):
            if _skip_part(p):
                continue
            txt = p.with_suffix(".txt")
            if not txt.is_file():
                continue
            text = txt.read_text(encoding="utf-8").strip().replace("\n", " ")
            if text:
                real.append((p, text))
    if os.environ.get("FGO_OCR_SYNTH", "1") == "0":
        synth = []
    if not real and not synth:
        raise SystemExit(f"沒有樣本。DATA={DATA} 先 python -m fgo_ocr synth")
    if real and synth:
        repeat = max(1, len(synth) // max(1, len(real)))
        real = real * repeat
    rows = synth + real
    print(f"samples synth={len(synth)} real_eff={len(real)} total={len(rows)} data={DATA}", flush=True)
    random.Random(7).shuffle(rows)
    return rows


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tune_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _autocast():
    if not torch.cuda.is_available():
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast()


def _grad_scaler():
    enabled = torch.cuda.is_available()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _fit_batch(model: nn.Module, device: torch.device) -> int:
    env = os.environ.get("FGO_OCR_BATCH", "").strip()
    if env:
        return int(env)
    if device.type != "cuda":
        return 16
    for b in (192, 160, 128, 96, 80, 64, 48, 32, 16):
        try:
            model.zero_grad(set_to_none=True)
            x = torch.randn(b, 3, IMG_H, MAX_W, device=device)
            y = torch.randint(1, 8, (b, 12), device=device)
            ylen = torch.full((b,), 12, dtype=torch.long, device=device)
            with _autocast():
                logits = model(x)
                logp = logits.log_softmax(2).permute(1, 0, 2)
                ilen = torch.full((b,), logp.size(0), dtype=torch.long, device=device)
                loss = nn.CTCLoss(blank=BLANK, zero_infinity=True)(logp, y, ilen, ylen)
            loss.backward()
            model.zero_grad(set_to_none=True)
            del x, y, logits, logp, loss
            torch.cuda.empty_cache()
            print(f"batch auto={b} vram={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB", flush=True)
            return b
        except Exception:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    return 8


def export_onnx(model: RecCRNN, path: Path, device: torch.device) -> None:
    model.eval()
    dummy = torch.randn(1, 3, IMG_H, MAX_W, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["x"],
        output_names=["logits"],
        opset_version=17,
    )


def _load_chars() -> str | None:
    for p in (OUT / "charset.json", MODELS / "charset.json"):
        if p.is_file():
            meta = json.loads(p.read_text(encoding="utf-8"))
            chars = meta.get("chars")
            if chars:
                return chars
    return None


def _ckpt_paths() -> list[Path]:
    return [
        OUT / "ctc.pt",
        MODELS / "rec.pt",
        MODELS / "exp_hires" / "ctc.pt",
    ]


def main() -> None:
    _tune_cuda()
    epochs = int(os.environ.get("FGO_OCR_EPOCHS", "120"))
    lr = float(os.environ.get("FGO_OCR_LR", "3e-4"))
    resume = os.environ.get("FGO_OCR_RESUME", "1") != "0"
    rows = _rows()
    kept = _load_chars() if resume else None
    if kept:
        chars = kept
        print(f"charset freeze n={len(chars)}", flush=True)
    else:
        texts = [t for _, t in rows] + load_labels()
        chars = build_charset(texts)
    table = {c: i + 1 for i, c in enumerate(chars)}
    nclass = len(chars) + 1
    device = _device()
    split = max(1, int(len(rows) * 0.97))
    train_set = RecSet(rows[:split], table)
    val_set = RecSet(rows[split:], table)
    workers = int(os.environ.get("FGO_OCR_WORKERS", "4" if os.name == "nt" else "8"))
    model = RecCRNN(nclass).to(device)
    saved = None
    if resume:
        for ck in _ckpt_paths():
            if not ck.is_file():
                continue
            saved = torch.load(ck, map_location="cpu")
            print(f"resume {ck}", flush=True)
            break
    if saved and saved.get("state"):
        missing, unexpected = model.load_state_dict(saved["state"], strict=False)
        if missing or unexpected:
            print(f"  load warn missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    batch = _fit_batch(model, device)
    dl_kw: dict = {
        "collate_fn": _collate,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, **dl_kw)
    val_kw = dict(dl_kw)
    val_kw["num_workers"] = max(0, workers // 2)
    if val_kw["num_workers"] <= 0:
        val_kw.pop("persistent_workers", None)
        val_kw.pop("prefetch_factor", None)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, **val_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.04)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    scaler = _grad_scaler()
    print(
        f"hires train={len(train_set)} val={len(val_set)} classes={nclass} "
        f"device={device} epochs={epochs} batch={batch} lr={lr} "
        f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}",
        flush=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y, ylen in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            ylen = ylen.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast():
                logits = model(x)
                logp = logits.log_softmax(2).permute(1, 0, 2)
                ilen = torch.full((x.size(0),), logp.size(0), dtype=torch.long, device=device)
                loss = ctc(logp, y, ilen, ylen)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            n += 1
        sched.step()
        model.eval()
        vtotal = 0.0
        vn = 0
        shown = None
        with torch.no_grad():
            for x, y, ylen in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                ylen = ylen.to(device, non_blocking=True)
                with _autocast():
                    logits = model(x)
                    logp = logits.log_softmax(2).permute(1, 0, 2)
                    ilen = torch.full((x.size(0),), logp.size(0), dtype=torch.long, device=device)
                    vtotal += float(ctc(logp, y, ilen, ylen).item())
                vn += 1
                if shown is None:
                    pred = ctc_greedy(logits[0].detach().float().cpu().numpy(), chars)
                    gold = "".join(chars[i - 1] for i in y[0, : ylen[0]].tolist() if i > 0)
                    shown = (gold, pred)
        tloss = total / max(1, n)
        vloss = vtotal / max(1, vn)
        g, p = shown or ("", "")
        print(
            f"epoch {ep}/{epochs} train={tloss:.3f} val={vloss:.3f} "
            f"lr={sched.get_last_lr()[0]:.2e} ex='{g}' -> '{p}'",
            flush=True,
        )
        ckpt = {
            "state": model.state_dict(),
            "chars": chars,
            "nclass": nclass,
            "img_h": IMG_H,
            "img_w": MAX_W,
            "arch": "crnn",
        }
        torch.save(ckpt, OUT / "ctc_last.pt")
        if vloss <= best:
            best = vloss
            torch.save(ckpt, OUT / "ctc.pt")
            (OUT / "charset.json").write_text(
                json.dumps(
                    {"chars": chars, "arch": "crnn", "img_h": IMG_H, "img_w": MAX_W},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            try:
                export_onnx(model, OUT / "ctc.onnx", device)
                print(f"  saved {OUT / 'ctc.onnx'} val={best:.3f}", flush=True)
            except Exception as e:
                print(f"  onnx export skip: {e}", flush=True)
    print("done", OUT / "ctc.onnx")


if __name__ == "__main__":
    main()
