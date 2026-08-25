from __future__ import annotations

import torch
from torch import nn

IMG_H = 32
IMG_W = 384
PATCH = 8
DIM = 256
ENC_LAYERS = 6
DEC_LAYERS = 2
HEADS = 8
MAX_LEN = 48
PAD, BOS, EOS = 0, 1, 2


class _Pos(nn.Module):
    def __init__(self, d: int, n: int):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, n, d))
        nn.init.trunc_normal_(self.p, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.p[:, : x.size(1)]


class PARSeq(nn.Module):
    def __init__(self, nclass: int):
        super().__init__()
        self.nclass = nclass
        self.max_len = MAX_LEN
        self.patch = nn.Conv2d(3, DIM, kernel_size=PATCH, stride=PATCH)
        enc_n = (IMG_H // PATCH) * (IMG_W // PATCH)
        self.enc_pos = _Pos(DIM, enc_n)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=DIM,
            nhead=HEADS,
            dim_feedforward=DIM * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=ENC_LAYERS)
        self.tok = nn.Embedding(nclass, DIM)
        self.dec_pos = _Pos(DIM, MAX_LEN + 1)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=DIM,
            nhead=HEADS,
            dim_feedforward=DIM * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=DEC_LAYERS)
        self.head = nn.Linear(DIM, nclass)
        self.drop = nn.Dropout(0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.patch(x)
        b, d, h, w = feat.shape
        z = feat.flatten(2).transpose(1, 2)
        return self.encoder(self.enc_pos(z))

    def _causal(self, t: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((t, t), float("-inf"), device=device), 1)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        t = tgt.size(1)
        q = self.drop(self.dec_pos(self.tok(tgt)))
        mask = self._causal(t, tgt.device)
        pad = tgt.eq(PAD)
        out = self.decoder(q, memory, tgt_mask=mask, tgt_key_padding_mask=pad)
        return self.head(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory = self.encode(x)
        b = x.size(0)
        ys = torch.full((b, 1), BOS, device=x.device, dtype=torch.long)
        steps = []
        for _ in range(self.max_len):
            logit = self.decode(ys, memory)[:, -1]
            steps.append(logit)
            ys = torch.cat([ys, logit.argmax(-1, keepdim=True)], dim=1)
        return torch.stack(steps, dim=1)


def greedy_text(logits: torch.Tensor, chars: str) -> str:
    ids = logits.argmax(-1).tolist()
    out: list[str] = []
    for i in ids:
        if i == EOS:
            break
        if i == PAD or i == BOS:
            continue
        j = i - 3
        if 0 <= j < len(chars):
            out.append(chars[j])
        s = "".join(out)
        for n in (8, 6, 4):
            if len(s) >= n * 2 and s[-n:] == s[-2 * n : -n]:
                return s[:-n]
    return "".join(out)


def encode_label(text: str, table: dict[str, int]) -> list[int]:
    ids = [BOS]
    for ch in text:
        if ch in table:
            ids.append(table[ch])
        if len(ids) >= MAX_LEN:
            break
    ids.append(EOS)
    return ids
