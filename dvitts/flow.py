"""Stage 2: content ids (50 Hz) + the voice's acoustic ids -> target acoustic ids (IndiCodec FSQ, 6 dims x 6 levels).
Flow matching on the 6 FSQ values per frame: 16 Euler steps on a sway schedule from a 0.7-scaled noise draw, classifier-free
guidance 4 (reference replaced by a learned null), then each value snaps to its nearest level."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_DIM, N_LEV = 6, 6


def ids_to_digits(ids: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """(..., ) acoustic ids -> (..., 6) level indices."""
    return (ids.long()[..., None] // basis.long()) % N_LEV


def digits_to_ids(d: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (d.long() * basis.long()).sum(-1)


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x (B, H, T, Dh), pos (B, T) -> rotated x."""
    dh = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    ang = pos.float()[:, None, :, None] * inv                       # (B, 1, T, Dh/2)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def level_values(n_lev: int = N_LEV) -> torch.Tensor:
    """vector-quantize-pytorch FSQ code of level index k: (k - L // 2) / (L // 2). For 6 levels: -1 .. 2/3."""
    hw = n_lev // 2
    return (torch.arange(n_lev, dtype=torch.float32) - hw) / hw


def digits_to_x(digits: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    return vals[digits.long().clamp(0, len(vals) - 1)]


def snap(x: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """(..., 6) values -> (..., 6) level indices (nearest level per dim)."""
    return (x[..., None] - vals).abs().argmin(-1)


def t_embed(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float() * 1000)[:, None] * f
    return torch.cat([a.cos(), a.sin()], -1)


def _in_dtype(mod):
    """Linear weight dtype, or None if quantized (dynamic INT8 exposes weight() as a method)."""
    w = getattr(mod, "weight", None)
    if w is None or callable(w):
        return None
    return w.dtype


class FlowBlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp: int = 4, drop: float = 0.1, qk_norm: bool = True):
        super().__init__()
        self.h, self.drop = heads, drop
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.qn, self.kn = (nn.LayerNorm(d // heads), nn.LayerNorm(d // heads)) if qk_norm else (None, None)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, mlp * d), nn.Linear(mlp * d, d)
        self.ada = nn.Linear(d, 6 * d)                              # AdaLN-Zero: zero-init in FlowStage2

    def forward(self, x, pos, pad, c):                              # c (B, d) time conditioning
        b, t, d = x.shape
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(c))[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        q, k, v = self.qkv(h).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if self.qn is not None:
            q, k = self.qn(q), self.kn(k)
        q, k = rope(q, pos), rope(k, pos)
        am = (~pad)[:, None, None, :] if pad is not None else None
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=self.drop if self.training else 0.0)
        x = x + g1 * F.dropout(self.proj(a.transpose(1, 2).reshape(b, t, d)), self.drop, self.training)
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * F.dropout(self.fc2(F.gelu(self.fc1(h))), self.drop, self.training)


class FlowStage2(nn.Module):
    flow = True

    def __init__(self, content_table: torch.Tensor, d: int = 768, layers: int = 12, heads: int = 12, drop: float = 0.1,
                 qk_norm: bool = True, meanflow: bool = False):
        super().__init__()
        self.meanflow = meanflow
        self.register_buffer("content_table", content_table.float(), persistent=True)   # (6561, 1024), frozen
        self.register_buffer("vals", level_values(), persistent=True)                    # (6,) FSQ level values
        self.c_in = nn.Linear(content_table.shape[1], d)
        self.x_in = nn.Linear(N_DIM, d)
        self.seg = nn.Embedding(2, d)
        self.null_ref = nn.Parameter(torch.zeros(d))
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.t_r_mix = nn.Linear(512, 256, bias=False) if meanflow else None            # Int-MeanFlow W: [I | 0]
        self.blocks = nn.ModuleList(FlowBlock(d, heads, drop=drop, qk_norm=qk_norm) for _ in range(layers))
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, N_DIM)
        self.apply(self._init)
        nn.init.normal_(self.null_ref, std=0.02)
        for blk in self.blocks:                                     # AdaLN-Zero: every block starts as identity
            nn.init.zeros_(blk.ada.weight)
            nn.init.zeros_(blk.ada.bias)
        for m in (self.ada_out, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        if self.t_r_mix is not None:
            with torch.no_grad():
                w = torch.zeros(256, 512)
                w[:, :256] = torch.eye(256)
                self.t_r_mix.weight.copy_(w)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, content, xv, seg, pad, t, drop, r=None):
        """content (B, T) ids at 50 Hz; xv (B, T, 6) [ref clean x1 | target x_t]; seg (B, T) 0 ref / 1 target;
        pad (B, T) True = padding or None; t (B,) in [0, 1]; drop (B,) bool = reference replaced by the null.
        r (B,) end of the MeanFlow interval, or None (= t, old instantaneous v).
        -> velocity (B, T, 6)."""
        dt = _in_dtype(self.x_in)
        x = self.c_in(self.content_table[content.long()]) + self.x_in(xv if dt is None else xv.to(dt))
        ref_null = drop[:, None] & (seg == 0)
        x = torch.where(ref_null[..., None], self.null_ref.to(x.dtype), x) + self.seg(seg)
        dt = _in_dtype(self.t_mlp[0])
        te = t_embed(t) if dt is None else t_embed(t).to(dt)
        if self.t_r_mix is not None:
            re = t_embed(t if r is None else r).to(te.dtype)
            te = self.t_r_mix(torch.cat([te, re], -1))
        c = self.t_mlp(te)
        pos = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
        for blk in self.blocks:
            x = blk(x, pos, pad, c)
        sh, sc = self.ada_out(F.silu(c))[:, None].chunk(2, -1)
        return self.out(self.norm(x) * (1 + sc) + sh)


def sway(n: int, device=None) -> torch.Tensor:
    u = torch.linspace(0, 1, n + 1, device=device)
    return 1 - torch.cos(u * math.pi / 2)


@torch.no_grad()
def generate_flow(model, content_ref, digits_ref, content_tgt, nfe: int = 16, w: float = 2.0,
                  gen: torch.Generator | None = None, w_until: float = 1.0, noise: float = 1.0) -> torch.Tensor:
    """One utterance. content_ref (P,), digits_ref (P, 6), content_tgt (T,) at 50 Hz -> target digits (T, 6).
    Euler on the sway schedule; CFG with cond and uncond (reference -> null) in one batch of 2 (w = 0: cond only).
    w_until < 1: guidance interval (Kynkaanniemi et al. 2024), CFG only on steps that start at t < w_until (the noisy
    steps that set the clip-level voice), cond only after (the late steps that set frame detail).
    noise < 1: start from a smaller draw of x0 (the sampling temperature of Matcha-TTS)."""
    dev = content_tgt.device
    p, t = len(content_ref), len(content_tgt)
    content = torch.cat([content_ref, content_tgt])[None]
    seg = torch.cat([torch.zeros(p, dtype=torch.long, device=dev), torch.ones(t, dtype=torch.long, device=dev)])[None]
    x_ref = digits_to_x(digits_ref, model.vals)
    x = torch.randn(t, N_DIM, device=dev, generator=gen) * noise
    drop2 = torch.tensor([False, True], device=dev)
    ts = sway(nfe, dev)
    for i in range(nfe):
        nb = 2 if w > 0 and float(ts[i]) < w_until else 1
        drop = drop2[:nb]
        xv = torch.cat([x_ref, x])[None].expand(nb, -1, -1)
        v = model(content.expand(nb, -1), xv, seg.expand(nb, -1), None, ts[i].expand(nb), drop)[:, p:].float()
        v = v[0] + w * (v[0] - v[1]) if nb == 2 else v[0]
        x = x + (ts[i + 1] - ts[i]) * v
    return snap(x, model.vals)
