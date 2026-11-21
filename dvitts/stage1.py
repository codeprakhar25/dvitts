"""Stage 1: text + voice prompt -> IndiCodec content ids (25 Hz), autoregressive, CosyVoice3 layout.

  [sos] [instruct + prompt text + target text] [task] [prompt content ids] -> sample until eos

The language model is a Qwen2 body (Fun-CosyVoice3-0.5B-2512 shape) with a speech embedding and a speech head.
`FastS1` is the serving loop: bf16 weights, a bare Qwen2 forward on preallocated KV buffers and a single-token step that
is captured as a CUDA graph (GPU) or traced (CPU). Sampling is CosyVoice's repetition-aware nucleus rule (top-p 0.8,
top-k 25; a draw that already fills the last 10 ids is redrawn from the full distribution) at temperature 0.8.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SPEECH = 6561                                  # content ids (= CosyVoice3 speech ids)
SOS, EOS, TASK = N_SPEECH + 0, N_SPEECH + 1, N_SPEECH + 2
NROWS = N_SPEECH + 200                           # CosyVoice3 speech table: 6561 ids + 200 special rows
INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
ENDOFPROMPT_ID = 151646


class S1LM(nn.Module):
    def __init__(self, body: nn.Module, d: int, rows: int = NROWS):
        super().__init__()
        self.body = body                                            # transformers Qwen2Model
        self.speech_embedding = nn.Embedding(rows, d)
        self.llm_decoder = nn.Linear(d, rows, bias=False)


def qwen2_body(cfg: dict):
    from transformers import Qwen2Config, Qwen2Model
    keys = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "max_position_embeddings", "rms_norm_eps", "rope_theta", "attention_dropout",
            "tie_word_embeddings", "hidden_act", "sliding_window", "use_sliding_window", "max_window_layers",
            "bos_token_id", "eos_token_id", "initializer_range")
    return Qwen2Model(Qwen2Config(**{k: cfg[k] for k in keys if k in cfg}))


def load_s1(sd: dict, cfg: dict) -> S1LM:
    rows, d = sd["speech_embedding.weight"].shape
    m = S1LM(qwen2_body(cfg), int(d), int(rows))
    m.load_state_dict(sd)
    return m.eval()


def rotate_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


class Bare:
    """Qwen2 forward on the model's own modules: no mask builder, no output dataclasses, KV in preallocated buffers."""

    def __init__(self, s1, max_len: int, dev):
        self.s1, self.body, self.dev = s1, s1.body, dev
        a0 = self.body.layers[0].self_attn
        self.hd, self.groups = a0.head_dim, a0.num_key_value_groups
        self.kvh = self.body.config.num_key_value_heads
        self.h = self.kvh * self.groups
        ref = self.body.norm.weight
        cos, sin = self.body.rotary_emb(ref.new_zeros(1, 1, self.body.config.hidden_size),
                                        torch.arange(max_len, device=dev)[None])
        self.cos, self.sin = cos[0].to(ref.dtype), sin[0].to(ref.dtype)

    def new_cache(self, b: int, length: int) -> tuple[list, list]:
        ref = self.body.norm.weight
        mk = lambda: [ref.new_zeros(b, self.kvh, length, self.hd) for _ in self.body.layers]  # noqa: E731
        return mk(), mk()

    def __call__(self, x, pos, n: int, K, V, mask=None, is_causal: bool = False):
        b, t, _ = x.shape
        cos, sin = self.cos[pos][:, None], self.sin[pos][:, None]
        for i, layer in enumerate(self.body.layers):
            at, h = layer.self_attn, layer.input_layernorm(x)
            q = at.q_proj(h).view(b, t, self.h, self.hd).transpose(1, 2)
            k = at.k_proj(h).view(b, t, self.kvh, self.hd).transpose(1, 2)
            v = at.v_proj(h).view(b, t, self.kvh, self.hd).transpose(1, 2)
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            K[i][:, :, n:n + t], V[i][:, :, n:n + t] = k, v
            T = n + t
            kk = K[i][:, :, :T][:, :, None].expand(b, self.kvh, self.groups, T, self.hd).reshape(b, self.h, T, self.hd)
            vv = V[i][:, :, :T][:, :, None].expand(b, self.kvh, self.groups, T, self.hd).reshape(b, self.h, T, self.hd)
            a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, is_causal=is_causal)
            x = x + at.o_proj(a.transpose(1, 2).reshape(b, t, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.body.norm(x)


class FixedStep(torch.nn.Module):
    """Single-token step over a fixed-length buffer and a position mask: one static shape, for a CUDA graph / trace."""

    def __init__(self, bare: Bare, K, V):
        super().__init__()
        self.body, self.head, self.emb = bare.body, bare.s1.llm_decoder, bare.s1.speech_embedding
        self.hd, self.groups, self.kvh, self.h = bare.hd, bare.groups, bare.kvh, bare.h
        self.register_buffer("cos", bare.cos, persistent=False)
        self.register_buffer("sin", bare.sin, persistent=False)
        for i, (k, v) in enumerate(zip(K, V)):
            self.register_buffer(f"k{i}", k, persistent=False)
            self.register_buffer(f"v{i}", v, persistent=False)

    def forward(self, idx, n_t, mask):
        x = self.emb(idx)[:, None]
        cos, sin = self.cos[n_t][None, None], self.sin[n_t][None, None]
        L = self.k0.shape[2]
        for i, layer in enumerate(self.body.layers):
            kb, vb = getattr(self, f"k{i}"), getattr(self, f"v{i}")
            at, h = layer.self_attn, layer.input_layernorm(x)
            q = at.q_proj(h).view(1, 1, self.h, self.hd).transpose(1, 2)
            k = at.k_proj(h).view(1, 1, self.kvh, self.hd).transpose(1, 2)
            v = at.v_proj(h).view(1, 1, self.kvh, self.hd).transpose(1, 2)
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            kb.index_copy_(2, n_t, k)
            vb.index_copy_(2, n_t, v)
            kk = kb[:, :, None].expand(1, self.kvh, self.groups, L, self.hd).reshape(1, self.h, L, self.hd)
            vv = vb[:, :, None].expand(1, self.kvh, self.groups, L, self.hd).reshape(1, self.h, L, self.hd)
            a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
            x = x + at.o_proj(a.transpose(1, 2).reshape(1, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.head(self.body.norm(x))[0, -1].float()


def ras_batch(logp, recent: torch.Tensor, gen, top_p=0.8, top_k=25, tau: int = 1) -> torch.Tensor:
    """Repetition-aware nucleus sampling: logp (B, V), recent (B, W) last ids or -1 -> ids (B,)."""
    p = logp.exp()
    vals, idx = p.topk(top_k, -1)
    keep = (vals.cumsum(-1) - vals) < top_p
    keep[:, 0] = True
    q = vals * keep
    j = idx.gather(1, torch.multinomial(q / q.sum(-1, keepdim=True), 1, generator=gen))[:, 0]
    bad = (recent == j[:, None]).sum(-1) >= tau
    if bad.any():
        j = j.clone()
        j[bad] = torch.multinomial(p[bad] / p[bad].sum(-1, keepdim=True), 1, generator=gen)[:, 0]
    return j


class FastS1:
    """One stream at a time. Casts the model to `dtype` in place. mode "graph" = CUDA graph (GPU default);
    "trace" = torch.jit.trace (CPU, or GPU fallback)."""

    def __init__(self, model, dev, max_len: int = 2048, dtype=torch.bfloat16, mode: str = "graph"):
        if mode not in ("trace", "graph"):
            raise ValueError(f"mode {mode}")
        if mode == "graph" and (str(dev) == "cpu" or not torch.cuda.is_available()):
            mode = "trace"
        self.m, self.dev, self.L, self.mode = model.to(dtype).eval(), dev, max_len, mode
        self.bare = Bare(self.m, max_len, dev)
        K, V = self.bare.new_cache(1, max_len)
        self.fs = FixedStep(self.bare, K, V).eval()
        self.K = [getattr(self.fs, f"k{i}") for i in range(len(K))]
        self.V = [getattr(self.fs, f"v{i}") for i in range(len(V))]
        self.fn = None

    def _prefill(self, x0: torch.Tensor) -> torch.Tensor:
        n = x0.shape[0]
        return self.bare(x0[None], torch.arange(n, device=self.dev)[None], 0, self.K, self.V, None, is_causal=True)[0, -1]

    def _prefix(self, text_ids, ref_speech) -> torch.Tensor:
        emb = self.m.speech_embedding.weight
        return torch.cat([emb[SOS][None], self.m.body.embed_tokens(torch.tensor(text_ids, device=self.dev)), emb[TASK][None],
                          self.m.speech_embedding(torch.tensor(ref_speech, dtype=torch.long, device=self.dev))])

    def _step_fn(self, n: int):
        if self.fn is None:
            mask = torch.zeros(1, 1, 1, self.L, dtype=torch.bool, device=self.dev)
            mask[..., : n + 1] = True
            ex = (torch.tensor([0], device=self.dev), torch.tensor([n], device=self.dev), mask)
            if self.mode == "graph":
                self.fn = self._cuda_graph(ex)
            else:
                self.fn = torch.jit.trace(self.fs, ex, check_trace=False)
                self.K = [getattr(self.fn, f"k{i}") for i in range(len(self.K))]
                self.V = [getattr(self.fn, f"v{i}") for i in range(len(self.V))]
        return self.fn

    def _cuda_graph(self, ex):
        idx, nt, mask = (t.clone() for t in ex)
        with torch.no_grad():                                       # not inference_mode: KV is updated in place on replay
            for _ in range(3):
                self.fs(idx, nt, mask)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.fs(idx, nt, mask)

        def run(idx_in, nt_in, mask_in):
            idx.copy_(idx_in)
            nt.copy_(nt_in)
            mask.copy_(mask_in)
            g.replay()
            return out.clone()

        return run

    @torch.no_grad()
    def generate(self, text_ids, ref_speech, n_target_text, gen, min_ratio=2.0, max_ratio=20.0, max_ids=None, top_p=0.8,
                 top_k=25, win=10, tau_r=0.1, temp=1.0) -> tuple[list[int], bool]:
        """-> (content ids, reached end of speech). End of speech is blocked before min_ratio x target text tokens."""
        N = N_SPEECH
        x0 = self._prefix(text_ids, ref_speech)
        n = x0.shape[0]
        lo, hi = int(n_target_text * min_ratio), max(int(n_target_text * max_ratio), 1)
        if max_ids:
            hi = min(hi, max(max_ids, lo + 1))
        hi = min(hi, self.L - n - 1)
        fn = self._step_fn(n)
        logits = self.m.llm_decoder(self._prefill(x0)).float()
        mask = torch.zeros(1, 1, 1, self.L, dtype=torch.bool, device=self.dev)
        mask[..., : n] = True
        recent = torch.full((1, win), -1, device=self.dev)
        tau = max(1, int(win * tau_r))
        out: list[int] = []
        for t in range(hi):
            logits[N + 2:] = -float("inf")                          # only speech ids and eos are valid outputs
            logits[N] = -float("inf")
            if t < lo:
                logits[EOS] = -float("inf")
            lp = torch.log_softmax(logits / temp, -1)[None]
            j = int(ras_batch(lp, recent, gen, top_p=top_p, top_k=top_k, tau=tau)[0])
            if j == EOS:
                return out, True
            out.append(j)
            recent = torch.cat([recent[:, 1:], torch.tensor([[j]], device=self.dev)], 1)
            mask[..., n] = True
            logits = fn(torch.tensor([j], device=self.dev), torch.tensor([n], device=self.dev), mask)
            n += 1
        return out, False
