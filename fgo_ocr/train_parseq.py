from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from fgo_ocr.dataset import build_charset, rows as load_rows
from fgo_ocr.parseq import (
    BOS,
    DIM,
    EOS,
    IMG_H,
    IMG_W,
    MAX_LEN,
    PAD,
    PARSeq,
    encode_label,
    greedy_text,
)
from fgo_ocr.paths import MODELS
from fgo_ocr.synth import load_labels


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecSet(Dataset):
    def __init__(self, items: list[tuple[Path, str]], table: dict[str, int]):
        self.rows = items
        self.table = table

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        path, text = self.rows[i]
        img = Image.open(path).convert("RGB")
        w = max(8, int(img.width * IMG_H / max(1, img.height)))
        w = min(IMG_W, w)
        img = img.resize((w, IMG_H), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (IMG_W, IMG_H), (20, 18, 16))
        canvas.paste(img, (0, 0))
        arr = np.asarray(canvas).astype(np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        y = torch.tensor(encode_label(text, self.table), dtype=torch.long)
        return x, y


def _collate(batch):
    xs, ys = zip(*batch)
    x = torch.stack(list(xs), 0)
    y = pad_sequence(list(ys), batch_first=True, padding_value=PAD)
    if y.size(1) < 2:
        y = torch.nn.functional.pad(y, (0, 2 - y.size(1)), value=EOS)
    return x, y


def export_onnx(model: PARSeq, path: Path, device: torch.device) -> None:
    model.eval()
    dummy = torch.randn(1, 3, IMG_H, IMG_W, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["x"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    epochs = int(os.environ.get("FGO_OCR_EPOCHS", "8"))
    batch = int(os.environ.get("FGO_OCR_BATCH", "16"))
    lr = float(os.environ.get("FGO_OCR_LR", "3e-4"))
    data = load_rows()
    texts = [t for _, t in data] + load_labels()
    chars = build_charset(texts)
    table = {c: i + 3 for i, c in enumerate(chars)}
    nclass = len(chars) + 3
    device = _device()
    resume = os.environ.get("FGO_OCR_RESUME", "0") == "1"
    ckpt_path = MODELS / "parseq.pt"
    saved = None
    if resume and ckpt_path.is_file():
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        chars = saved.get("chars") or chars
        nclass = int(saved.get("nclass") or nclass)
        table = {c: i + 3 for i, c in enumerate(chars)}
        print(f"resume {ckpt_path} classes={nclass}", flush=True)
    split = max(1, int(len(data) * 0.9))
    train_loader = DataLoader(
        RecSet(data[:split], table),
        batch_size=batch,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        RecSet(data[split:], table),
        batch_size=batch,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )
    model = PARSeq(nclass).to(device)
    if saved and saved.get("state"):
        missing, unexpected = model.load_state_dict(saved["state"], strict=False)
        if missing or unexpected:
            print(f"  load warn missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(ignore_index=PAD)
    print(
        f"parseq train={len(data[:split])} val={len(data[split:])} "
        f"classes={nclass} dim={DIM} device={device} epochs={epochs}",
        flush=True,
    )
    MODELS.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            memory = model.encode(x)
            tgt_in = y[:, :-1]
            tgt_out = y[:, 1:]
            logits = model.decode(tgt_in, memory)
            loss = ce(logits.reshape(-1, nclass), tgt_out.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            n += 1
        model.eval()
        vtotal = 0.0
        vn = 0
        shown = None
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model.decode(y[:, :-1], model.encode(x))
                vtotal += float(ce(logits.reshape(-1, nclass), y[:, 1:].reshape(-1)).item())
                vn += 1
                if shown is None:
                    pred = greedy_text(model(x[:1])[0], chars)
                    gold_ids = y[0].tolist()
                    gold = "".join(
                        chars[i - 3]
                        for i in gold_ids
                        if i >= 3 and i - 3 < len(chars)
                    )
                    shown = (gold, pred)
        tloss = total / max(1, n)
        vloss = vtotal / max(1, vn)
        g, p = shown or ("", "")
        print(f"epoch {ep}/{epochs} train={tloss:.3f} val={vloss:.3f} ex='{g}' -> '{p}'", flush=True)
        if vloss < best:
            best = vloss
            torch.save(
                {"state": model.state_dict(), "chars": chars, "nclass": nclass, "arch": "parseq"},
                MODELS / "parseq.pt",
            )
            (MODELS / "charset.json").write_text(
                json.dumps(
                    {
                        "chars": chars,
                        "arch": "parseq",
                        "img_h": IMG_H,
                        "img_w": IMG_W,
                        "max_len": MAX_LEN,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            try:
                export_onnx(model, MODELS / "parseq.onnx", device)
                print(f"  saved {MODELS / 'parseq.onnx'} val={best:.3f}", flush=True)
            except Exception as e:
                print(f"  onnx export skip: {e}", flush=True)
    print("done", MODELS / "parseq.onnx")


if __name__ == "__main__":
    main()