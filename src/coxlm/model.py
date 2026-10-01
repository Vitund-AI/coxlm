"""The amortized decision model: encoder + schema-built readout, one forward pass.

Mechanically this is a bidirectional encoder with a cross-attention readout head.
Each output field becomes a query that pools the encoded state; each field's
answer is scored by comparing that pooled slot against the embeddings of the
field's allowed options (a bi-encoder, late-interaction comparison). Every field
is produced in the same parallel pass, so the fields are conditionally
independent given the input state. Any dependency between decisions has to live
in the code around the model, not inside it -- that is the central bet, and its
main limitation.

The learnable parts beyond the encoder are small: one cross-attention block, one
projection, and a scalar temperature. Everything else is the pretrained encoder,
which is why this is a fine-tuning/distillation job rather than a from-scratch
training job.

coxlm ships this module for inference only; training lives in the research
repository.
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decide import Answer, answers_for, render_state
from .schema import Schema


class ReadoutLayer(nn.Module):
    """One readout step: every field query cross-attends to the state tokens,
    then passes through a feed-forward block. Post-norm, like the single-layer
    readout it generalises. There is deliberately no attention between the
    queries, so a field's answer never depends on which other fields were asked
    in the same request."""

    def __init__(self, d: int, heads: int, ff_mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, queries: torch.Tensor, hidden: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(queries, hidden, hidden, key_padding_mask=key_padding_mask)
        queries = self.norm1(queries + a)
        return self.norm2(queries + self.ff(queries))


class OptionBlockLayer(nn.Module):
    """One set-relative step: a question's slot and its options attend to each other
    (full attention within the field, nothing across fields), then a feed-forward
    block. Pre-norm, and both output projections start at zero, so a model with
    this block begins exactly as the model without it and learns what to add.

    The plain head scores each option against the state in isolation, so an option
    cannot know what else is on offer: a never-seen option beside none_of_these
    wins by default, and "none of these" cannot mean "none of THESE". Here the
    options are compared with each other, in view of the state, before scoring."""

    def __init__(self, d: int, heads: int, ff_mult: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ff[2].weight)
        nn.init.zeros_(self.ff[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1 + k, d)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.ff(self.norm2(x))


class AmortizedDecisionModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        tokenizer,
        hidden_size: int,
        proj_dim: int | None = None,
        num_heads: int = 4,
        freeze_encoder: bool = True,
        max_length: int = 256,
        normalize: bool = True,
        init_temp: float | None = None,
        pool: str = "mean",
        state_norm: bool | str = False,
        readout_layers: int = 1,
        question_in_state: bool = False,
        question_mode: str | None = None,
        packed_state_attention: str = "native",
        packed_impl: str = "auto",
        read_layer: int | None = None,
        option_block: int = 0,
        prompt_format: str = "xml",
        task_meta_pos: str = "off",
        span_pool: str = "last",
        answer_sentinel: str = "none",
        pause_tokens: int = 0,
        pause_mode: str = "learned",
        lora_scope: str = "all",
        recycle: int = 0,
    ) -> None:
        """``pool`` is how field and option texts are collapsed into one vector:
        "mean" over tokens (bidirectional encoders) or "last" token (causal
        decoders used as encoders, where only the final position has seen the
        whole text). The state itself is never pooled; the readout attends over
        its tokens either way."""
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.max_length = max_length
        self._encoder_frozen = freeze_encoder
        self.normalize = normalize
        if pool not in ("mean", "last"):
            raise ValueError("pool must be 'mean' or 'last'")
        self.pool = pool
        # Encoder hidden states carry a few "massive activation" dimensions that
        # hold most of the energy (measured: 4 of 768 dims hold 53% in
        # ModernBERT-base, and any two random token states have cosine 0.55;
        # 0.83 in Qwen3-1.7B). Cosine scores and dot-product attention then see
        # mostly those dimensions. A LayerNorm does not help: it rescales the
        # vector and leaves its direction alone. Standardising each dimension
        # with fixed statistics does (random-token cosine -> 0.00).
        #   "standardize"  per-dimension (h - mu) / sd, calibrated from data once
        #   "layer" / True LayerNorm (kept for old checkpoints)
        #   False          nothing (so existing checkpoints keep their keys)
        mode = {True: "layer", False: "off", None: "off"}.get(state_norm, state_norm)
        if mode not in ("off", "layer", "standardize"):
            raise ValueError("state_norm must be off / layer / standardize")
        self.state_norm_mode = mode
        self.state_norm = nn.LayerNorm(hidden_size) if mode == "layer" else None
        if mode == "standardize":
            self.register_buffer("state_mu", torch.zeros(hidden_size))
            self.register_buffer("state_sd", torch.ones(hidden_size))
            self.register_buffer("state_calibrated", torch.zeros((), dtype=torch.bool))

        proj = proj_dim or hidden_size
        # question_in_state turns the bi-encoder into a cross-encoder: each
        # field's question is prepended to the state and the encoder runs once
        # per field, so the state is encoded in light of the question. It costs
        # F passes instead of one and gives up the cached single state pass; it
        # exists to test whether unseen questions fail because the state never
        # saw the question. The readout, options and scoring are unchanged.
        # question_mode names where the question meets the state:
        #   "late"    the published design: state encoded alone, once; each field
        #             query reads it through the cross-attention readout.
        #   "cross"   question prepended, full bidirectional attention, one pass
        #             per field: the state is encoded in light of the question.
        #   "packed"  one pass over [state | q1 | q2 | ...] under a block mask:
        #             each question reads the state (and itself) at every layer,
        #             the state never reads a question, questions never read
        #             each other; question positions restart after the state.
        #             Each field's slot is read from its own question tokens.
        if question_mode is None:
            question_mode = "cross" if question_in_state else "late"
        if question_mode not in ("late", "cross", "packed"):
            raise ValueError("question_mode must be late / cross / packed")
        self.question_mode = question_mode
        self.question_in_state = question_mode == "cross"
        # packed mode on a causal backbone: "native" keeps the state block causal
        # (what the model was pretrained on, no distribution shift);
        # "bidirectional" lets state tokens read the whole state (a stronger state
        # encoding, still shareable across questions, but a shift the fine-tune
        # has to absorb). Bidirectional encoders always read the state both ways.
        if packed_state_attention not in ("native", "bidirectional"):
            raise ValueError("packed_state_attention must be native / bidirectional")
        self.packed_state_attention = packed_state_attention
        # How packed mode keeps questions from reading each other:
        #   "mask"       one sequence [state | q1 | q2 ...] under a block attention mask (the original)
        #   "replicate"  one row per question, [state | q_i], under the backbone's own causal mask
        # Both give each question the state and itself only. "replicate" works for backbones that cannot
        # take a custom mask -- recurrent / linear-attention layers (Qwen3.5, hybrid Mamba) scan the whole
        # sequence -- at the cost of encoding the state once per question. "auto" picks it for those.
        if packed_impl not in ("auto", "mask", "replicate"):
            raise ValueError("packed_impl must be auto / mask / replicate")
        if packed_impl == "auto":
            lt = set(getattr(encoder.config, "layer_types", None) or [])
            hybrid = bool(lt & {"linear_attention", "mamba", "recurrent"}) or getattr(encoder.config, "model_type", "") in HYBRID_MODEL_TYPES
            packed_impl = "replicate" if hybrid else "mask"
        self.packed_impl = packed_impl
        # Which layer of the backbone the head reads: None is the final layer (after the
        # backbone's final norm); k reads hidden_states[k], the residual stream after block
        # k (0 is the embeddings). The final layer of a decoder is shaped for predicting
        # the next token, which is not what the head needs; an earlier layer may fit
        # better, and every block above it can then be dropped (``truncate_backbone``).
        self.read_layer = read_layer
        # How the question, its options and (in packed mode) the state are written as
        # text: "plain" is labelled lines ("question: ...", "options: a: ...; b: ..."),
        # "xml" wraps each in tags. A checkpoint only reads the format it was trained in.
        if prompt_format not in ("plain", "xml"):
            raise ValueError(f"unknown prompt_format {prompt_format!r}")
        self.prompt_format = prompt_format
        # xml only: the tag naming the state block, and text prefixed to every question's instructions (both tests
        # of whether the question needs a named referent for the data; saved in the checkpoint like the format)
        self.state_tag = "state"
        self.question_prefix = ""
        self.state_preamble = ""  # a constant sentence before the state block (e.g. declaring the text's role)
        # question-aware reading: each question's block (without the <answer> marker) ALSO placed before the state,
        # so the state is read knowing what will be asked. Replicated layout only (one row per question anyway).
        # COX_QUESTION_FIRST=1 turns it on at load time, for evaluating existing models.
        self.question_first = os.environ.get("COX_QUESTION_FIRST", "") not in ("", "0")
        # "slot": score the <answer> vector against each option encoded ON ITS OWN (the published design).
        # "pointer": score it against each option's hidden state IN CONTEXT -- the last token of that option's line in
        # the question block, which has read the state and the question (pointer readout; set_readout()).
        self.readout_mode = "slot"
        self.pointer_q = self.pointer_k = None
        # pointer mode only: each option line sees the state, its question and itself (not the other options), all
        # option lines start at the same position, and the tokens after them (</options>, <answer>) see every option
        # at shared positions -- option-order-free by construction.
        self.option_isolation = False
        # pointer mode: for questions with at most this many options, each option's description ends with the other
        # options' labels, "(as opposed to: b, c)", sorted by name so presentation order still cannot matter. 0 = off.
        self.option_contrast = 0
        # pointer mode: False = yes/no fields are scored by the slot readout instead (hybrid: pointer for choice and
        # score fields, where it wins; slot for yes/no, where it was strongest in every run)
        self.pointer_yesno = True
        # with pointer_yesno False: "pointer" keeps the pointer layout (all options written, isolation applied) for yes/no
        # questions and only scores them with the slot readout (the first hybrid); "slot" renders them exactly as the
        # slot readout does -- plain question block, no isolation -- so yes/no questions are what the control sees.
        self.yesno_layout = "pointer"
        # "letters" readout: each option line starts with a one-token code (A, B, ... AA, ...), inserted as its
        # exact token id so it can never merge with neighbouring text; the <answer> state is scored against the language
        # model's own output rows for those codes (tied embeddings, so the readout starts as the base model's letter
        # prediction). letter_codes "custom": the input side uses learned marker vectors initialised from the letter
        # embeddings (placeholders CODE_BASE - j) instead of the letter tokens themselves.
        self.letter_codes = "letters"
        self.code_ids: list[int] = []
        self.code_readout = None
        self.code_emb = None
        if task_meta_pos not in ("off", "state_front", "state_end", "block", "stem"):
            raise ValueError(f"unknown task_meta_pos {task_meta_pos!r}")
        self.task_meta_pos = task_meta_pos
        if span_pool not in ("last", "mean", "attn"):
            raise ValueError(f"unknown span_pool {span_pool!r}")
        # how a causal backbone reads each question span: the last token ("last", the
        # <answer> position) or the mean over the whole span ("mean"). Bidirectional
        # encoders always mean-pool.
        self.span_pool = span_pool
        if answer_sentinel not in ("none", "eos"):
            raise ValueError(f"unknown answer_sentinel {answer_sentinel!r}")
        # "eos": append the EOS token to each question block so the last-token tap reads a
        # position the base model was pretrained to summarize at (vs the generic '>' of <answer>).
        self.answer_sentinel = answer_sentinel
        self.pause_emb = None
        self.set_pause(pause_tokens, pause_mode)
        # "all": LoRA adapts every token (the published design). "questions": LoRA is masked off at state positions,
        # so the state is encoded by the frozen base model and only question tokens are adapted -- one state
        # encoding then serves any question set, any adapter and text generation from the same cache.
        if lora_scope not in ("all", "questions"):
            raise ValueError(f"unknown lora_scope {lora_scope!r}")
        self.lora_scope = lora_scope
        self._tok_mask = None  # (rows, n) 1 = adapt this token; set around backbone calls when lora_scope == "questions"
        self._lora_hooks = []
        self.recycle_proj = None
        self.set_recycle(recycle)
        # readout_layers == 1 is the published design: one cross-attention from
        # each field query to the state tokens. More layers let a question refine
        # what it reads from the state in several steps, at the cost of a few
        # million parameters; the state is still encoded exactly once.
        self.readout_layers = readout_layers
        if readout_layers <= 1:
            self.readout = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            self.readout_norm = nn.LayerNorm(hidden_size)
            self.readout_stack = None
        else:
            self.readout_stack = nn.ModuleList(ReadoutLayer(hidden_size, num_heads) for _ in range(readout_layers))
        self.proj = nn.Linear(hidden_size, proj, bias=False)
        # learned attention-pool over a question span (used only when span_pool=="attn");
        # created always so the state_dict is consistent across span_pool settings.
        self.span_attn = nn.Linear(hidden_size, 1)
        # option_block > 0: that many set-relative layers between projection and scoring
        self.option_block = nn.ModuleList(OptionBlockLayer(proj, 8 if proj % 8 == 0 else num_heads) for _ in range(option_block)) if option_block else None
        # Scores are cosine similarities when ``normalize`` is on, so they live in
        # [-1, 1] and the temperature sets the logit scale directly (CLIP-style,
        # 0.07 -> logits in about [-14, 14]). Raw dot products of pooled encoder
        # states are in the hundreds, which pins every softmax at one-hot from
        # step zero and leaves a temperature initialised near 1.0 unable to catch
        # up; the head then has to fix the scale by shrinking its own weights.
        if init_temp is None:
            init_temp = 0.07 if normalize else 1.0
        # temperature = softplus(log_temp) + eps; invert softplus for the init
        self.log_temp = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(init_temp)))))

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        # schema embeddings are cached only when the encoder is frozen, since a
        # trainable encoder must recompute them each step to receive gradient.
        self._cache: dict = {}

    # -- small helpers -----------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.proj.weight.device

    # -- pause tokens: extra positions after each question, before the readout ---------
    # K positions appended to every question block (after <answer>, and after any sentinel); the answer is read
    # from the last of them. Each reads the state, its own question and the earlier pauses (the same isolation as
    # the question); nothing else ever reads them, so they cannot change any other token. "learned": K trainable
    # vectors (Goyal et al.'s pause tokens); "token:<text>": K copies of an existing token (no new parameters) --
    # the control that separates extra positions from learned vectors.
    PAUSE_BASE = -1000  # placeholder ids for learned pauses: PAUSE_BASE - j for pause j

    def set_pause(self, k: int, mode: str = "learned") -> None:
        if k < 0 or not (mode == "learned" or mode.startswith("token:")):
            raise ValueError(f"bad pause setting: k={k}, mode={mode!r}")
        self.pause_tokens, self.pause_mode = k, mode
        self.pause_emb = None
        if k and mode == "learned":
            w = self.encoder.get_input_embeddings().weight.detach().float()
            init = w.mean(0, keepdim=True) + w.std(0, keepdim=True) * torch.randn(k, w.size(1), device=w.device) * 0.1
            self.pause_emb = nn.Parameter(init)
        self._cache.clear() if hasattr(self, "_cache") else None

    def _pause_ids(self) -> list[int]:
        if not self.pause_tokens:
            return []
        if self.pause_mode == "learned":
            return [self.PAUSE_BASE - j for j in range(self.pause_tokens)]
        ids = self.tokenizer(self.pause_mode[len("token:"):], add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"pause token {self.pause_mode!r} is {len(ids)} tokens; pick a single-token string")
        return ids * self.pause_tokens

    def install_lora_scope(self) -> None:
        """Scale every LoRA update by the current token mask (hooks on the lora_B projections). Called once after LoRA
        is attached, for lora_scope == 'questions'; a no-op otherwise."""
        for h in self._lora_hooks:
            h.remove()
        self._lora_hooks = []
        if self.lora_scope != "questions":
            return

        def hook(_mod, _inp, out):
            m = self._tok_mask
            return out if m is None or m.shape != out.shape[:2] else out * m.unsqueeze(-1).to(out.dtype)

        for name, mod in self.encoder.named_modules():
            if name.split(".")[-2:-1] == ["lora_B"] or (name.endswith("lora_B.default")):
                self._lora_hooks.append(mod.register_forward_hook(hook))
        if not self._lora_hooks:
            raise RuntimeError("lora_scope='questions' needs LoRA adapters (lora_r > 0)")

    # -- answer recycling: feed each question's own answer estimate back in and re-answer ------------------------
    # One ESTIMATE token after each question block (after any pauses); its input vector is the probability-weighted
    # mix of the question's option embeddings (scaled to input-embedding size, through a learned projection that
    # starts as the identity), and the answer is read from it. Round 0 uses a uniform estimate; each later round feeds
    # back the previous round's distribution. Training (AlphaFold-2 recycling): n ~ U{0..N} rounds without gradient,
    # then one graded round -- the model only ever sees its own estimates, never the gold.
    RECYCLE_ID = -500

    def set_readout(self, mode: str, dp: int = 256) -> None:
        """'slot' (default) or 'pointer' (see readout_mode). Pointer mode adds a bilinear head (dp = its width)."""
        if mode not in ("slot", "pointer", "letters"):
            raise ValueError(f"unknown readout mode {mode!r}")
        self.readout_mode = mode
        self.pointer_q = self.pointer_k = None
        if mode == "letters":
            self._init_letter_codes()
        if mode == "pointer":
            if self.question_mode != "packed" or self.option_block is not None:
                raise NotImplementedError("pointer readout needs question_mode='packed' and no option block")
            d = self.encoder.get_input_embeddings().weight.shape[1]
            dev = self.encoder.get_input_embeddings().weight.device
            self.pointer_q, self.pointer_k = nn.Linear(d, dp).to(dev), nn.Linear(d, dp).to(dev)
            self._pointer_scale = dp ** -0.5
        if hasattr(self, "_cache"):
            self._cache.clear()

    CODE_BASE = -100  # placeholder ids for learned option markers: CODE_BASE - j for code j (pause <= -1000, recycle -500)

    def _init_letter_codes(self, n_max: int = 255) -> None:
        import itertools
        import string
        if self.question_mode != "packed":
            raise NotImplementedError("letter readout needs question_mode='packed'")
        tok = self.tokenizer
        cands = list(string.ascii_uppercase) + ["".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)]
        ids = []
        for c in cands:
            enc = tok.encode(c, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
            if len(ids) == n_max:
                break
        self.code_ids = ids
        emb = self.encoder.get_input_embeddings().weight
        with torch.no_grad():
            rows = emb[torch.tensor(ids, device=emb.device)].detach().float().clone()
        self.code_readout = nn.Parameter(rows)  # (n_codes, d); tied embeddings -> the base model's letter logits
        self.code_emb = nn.Parameter(rows.clone()) if self.letter_codes == "custom" else None
        if hasattr(self, "_cache"):
            self._cache.clear()

    def set_letter_codes(self, kind: str) -> None:
        if kind not in ("letters", "custom"):
            raise ValueError(f"unknown letter codes {kind!r}")
        self.letter_codes = kind
        if self.readout_mode == "letters":
            self._init_letter_codes()

    def _letters_question_block(self, f) -> list[int]:
        """Question block with a one-token code at the start of each option line, inserted as its exact id."""
        instr = f"{self.question_prefix}{f.instructions}" if f.instructions else self.question_prefix.strip()
        head = (f'\n<question name="{f.name}">{instr}</question>' if instr else f'\n<question name="{f.name}"/>') + "\n<options>"
        enc = lambda t: self._plain_ids([t], 100_000)[0]  # noqa: E731
        lines = self._option_texts(f)
        if len(lines) > len(self.code_ids):
            raise ValueError(f"{len(lines)} options exceed the {len(self.code_ids)} letter codes")
        ids = enc(head)
        for j, line in enumerate(lines):
            code = self.code_ids[j] if self.letter_codes == "letters" else self.CODE_BASE - j
            ids += enc("\n") + [code] + enc(": " + line)
        return ids + enc("\n</options>\n<answer>")

    def set_recycle(self, n: int) -> None:
        """n > 0: up to n recycled rounds. n == -1: the estimate token with the uniform estimate only, never recycled
        (control: an option-summary token without feedback). 0: off."""
        self.recycle_token = int(n) != 0
        self.recycle_rounds = max(int(n), 0)
        self.eval_rounds = self.recycle_rounds
        self.recycle_proj = None
        self._recycle_vecs = None
        self.last_round_probs = None
        if self.recycle_token:
            d = self.encoder.get_input_embeddings().weight.shape[1]
            self.recycle_proj = nn.Linear(d, d, bias=True)
            with torch.no_grad():
                self.recycle_proj.weight.copy_(torch.eye(d))
                self.recycle_proj.bias.zero_()
            self.recycle_proj.to(self.encoder.get_input_embeddings().weight.device)
            w = self.encoder.get_input_embeddings().weight.detach().float()
            self._emb_norm = float(w.norm(dim=-1).mean())
        if hasattr(self, "_cache"):
            self._cache.clear()

    def _estimate_vectors(self, probs: list[torch.Tensor], option_embs: list[torch.Tensor]) -> torch.Tensor:
        """probs: per field (B, n_f) -> (B*F, d) estimate vectors in row-major (b, field) order."""
        vecs = []
        for p, o in zip(probs, option_embs):  # p (B, n_f), o (n_f, d)
            v = p.to(o.dtype) @ o[: p.size(1)]
            vecs.append(F.normalize(v.float(), dim=-1) * self._emb_norm)
        v = torch.stack(vecs, dim=1).reshape(-1, vecs[0].size(-1))  # (B*F, d)
        return self.recycle_proj(v.to(self.recycle_proj.weight.dtype))

    def _backbone_inputs(self, ids: torch.Tensor) -> dict:
        """input_ids, or inputs_embeds with learned pause vectors / recycled estimates written in at their placeholders."""
        has_p = self.pause_emb is not None and bool((ids <= self.PAUSE_BASE).any())
        has_r = self._recycle_vecs is not None and bool((ids == self.RECYCLE_ID).any())
        is_c = (ids <= self.CODE_BASE) & (ids > self.CODE_BASE - 255)  # code placeholders -100 .. -354
        has_c = self.code_emb is not None and bool(is_c.any())
        if not (has_p or has_r or has_c):
            return {"input_ids": ids.clamp(min=0) if bool((ids < 0).any()) else ids}
        is_p = ids <= self.PAUSE_BASE
        is_r = ids == self.RECYCLE_ID
        emb = self.encoder.get_input_embeddings()(ids.masked_fill(is_p | is_r | is_c, 0))
        if has_c:
            j = (self.CODE_BASE - ids).clamp(min=0, max=self.code_emb.size(0) - 1)
            emb = torch.where(is_c.unsqueeze(-1), self.code_emb.to(emb.dtype)[j], emb)
        if has_p:
            j = (self.PAUSE_BASE - ids).clamp(min=0)
            emb = torch.where(is_p.unsqueeze(-1), self.pause_emb.to(emb.dtype)[j.clamp(max=self.pause_tokens - 1)], emb)
        if has_r:
            flat = emb.reshape(-1, emb.size(-1)).clone()
            idx = is_r.reshape(-1).nonzero(as_tuple=True)[0]  # row-major: matches (b, field) order
            if idx.numel() != self._recycle_vecs.size(0):
                raise RuntimeError(f"recycle placeholders {idx.numel()} != estimate vectors {self._recycle_vecs.size(0)}")
            flat[idx] = self._recycle_vecs.to(flat.dtype)
            emb = flat.view_as(emb)
        return {"inputs_embeds": emb}

    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temp) + 1e-4

    def truncate_backbone(self) -> int:
        """Drop every backbone block above ``read_layer``. The head's input does not
        change (the final norm goes too, since hidden_states[k] is the residual stream
        before it), so answers are identical and every decision costs k / n of the
        backbone. Returns the number of blocks removed."""
        if self.read_layer is None:
            return 0
        k = self.read_layer
        for m in self.encoder.modules():
            layers = getattr(m, "layers", None)
            if isinstance(layers, nn.ModuleList) and hasattr(m, "norm"):
                n = len(layers)
                k = k if k >= 0 else n + 1 + k
                if not 0 < k <= n:
                    raise ValueError(f"read_layer {self.read_layer} is outside 1..{n}")
                if k == n:
                    return 0  # hidden_states[n] is the normed final layer; nothing to drop
                m.layers = nn.ModuleList(list(layers)[:k])
                m.norm = nn.Identity()
                for cfg in {id(c): c for c in (getattr(m, "config", None), getattr(self.encoder, "config", None)) if c is not None}.values():
                    cfg.num_hidden_layers = k
                    if getattr(cfg, "layer_types", None):
                        cfg.layer_types = list(cfg.layer_types)[:k]
                self.read_layer = None  # the last hidden state is now the layer that was read
                self.clear_cache()
                return n - k
        raise NotImplementedError("could not find the backbone's block list to truncate")

    def _backbone_hidden(self, **kw) -> torch.Tensor:
        """Run the backbone and return the hidden states of the layer the head reads,
        in the head's dtype, standardised if the model is set to."""
        thr = getattr(self, "ckpt_min_tokens", 0)
        if thr and self.training:  # checkpoint only big inputs: short batches keep activations (no recompute cost)
            ids = kw.get("input_ids", kw.get("inputs_embeds"))
            want = ids is not None and ids.shape[0] * ids.shape[1] >= thr
            if want != getattr(self, "_ckpt_on", True):
                (self.encoder.gradient_checkpointing_enable if want else self.encoder.gradient_checkpointing_disable)()
                self._ckpt_on = want
        if self.read_layer is None:
            out = self.encoder(**kw)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        else:
            hidden = self.encoder(**kw, output_hidden_states=True).hidden_states[self.read_layer]
        # the encoder may run in bf16 (large decoders); the head stays fp32
        hidden = hidden.to(self.proj.weight.dtype)
        if self.readout_mode == "letters":  # the LM-head rows expect the backbone's own final hidden state
            return hidden
        if self.state_norm is not None:
            hidden = self.state_norm(hidden)
        elif self.state_norm_mode == "standardize":
            hidden = (hidden - self.state_mu) / self.state_sd
        return hidden

    def _encode_texts(self, texts: list[str], bidirectional: bool = False):
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        if bidirectional and self.pool == "last":
            # a causal backbone asked to read bidirectionally: every real token sees every real token
            allow = attn.bool().unsqueeze(1) & attn.bool().unsqueeze(2)
            allow = allow | torch.eye(attn.size(1), dtype=torch.bool, device=attn.device).unsqueeze(0)
            hidden = self._backbone_hidden(input_ids=input_ids, attention_mask=self._additive(allow))
        else:
            hidden = self._backbone_hidden(input_ids=input_ids, attention_mask=attn)
        return hidden, attn

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        mask = attn.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(1)
        counts = mask.sum(1).clamp(min=1.0)
        return summed / counts

    @staticmethod
    def _last_pool(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """Hidden state at each sequence's last non-pad token (works for either padding side)."""
        if bool((attn[:, -1] == 1).all()):  # left padding: last column is always real
            return hidden[:, -1, :]
        idx = attn.sum(1).clamp(min=1) - 1  # right padding
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx, :]

    def _pool(self, hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        return self._last_pool(hidden, attn) if self.pool == "last" else self._mean_pool(hidden, attn)

    def _schema_embeddings(self, schema: Schema, cache_ok: bool):
        """Return (field_query [F, d], [option_emb per field [k_f, d]])."""
        # keyed on the schema's content (it is frozen and hashable), not its object
        # id: ids are recycled after garbage collection, and a stale hit returns
        # another schema's embeddings
        if len(self._cache) > 512:
            self._cache.clear()
        if cache_ok and schema in self._cache:
            return self._cache[schema]

        # the question in words (name, or "name: instructions") and each option
        # with its description are read by the same encoder as the state
        field_hidden, field_attn = self._encode_texts([f.query_text() for f in schema])
        field_q = self._pool(field_hidden, field_attn)  # (F, d)

        option_embs = []
        for f in schema:
            oh, oa = self._encode_texts(self._option_texts(f))
            option_embs.append(self._pool(oh, oa))  # (k_f, d)

        if cache_ok:
            cached = (field_q.detach(), [o.detach() for o in option_embs])
            self._cache[schema] = cached
            return cached
        return field_q, option_embs

    # -- packed questions: one pass, block attention mask -------------------

    def _additive(self, allow: torch.Tensor) -> torch.Tensor:
        """(B, n, n) bool 'may attend' -> (B, 1, n, n) additive mask in the encoder's dtype."""
        dtype = next(self.encoder.parameters()).dtype
        mask = torch.zeros(allow.shape, dtype=dtype, device=allow.device)
        mask.masked_fill_(~allow, torch.finfo(dtype).min)
        return mask.unsqueeze(1)

    def _plain_ids(self, texts: list[str], max_length: int) -> list[list[int]]:
        """Token ids per text, memoised: training re-reads the same states (and, under channel ablation, many of the
        same question strings) every epoch, so each text is tokenised once. Returns fresh lists (callers extend them)."""
        cache = self.__dict__.setdefault("_tok_memo", {})
        if len(cache) > 2_000_000:
            cache.clear()
        missing = list(dict.fromkeys(t for t in texts if (t, max_length) not in cache))
        if missing:
            enc = self.tokenizer(missing, add_special_tokens=False, truncation=True, max_length=max_length, padding=False, return_tensors=None)
            for t, ids in zip(missing, enc["input_ids"]):
                cache[(t, max_length)] = tuple(ids)
        return [list(cache[(t, max_length)]) for t in texts]

    def _option_texts(self, f) -> list[str]:
        """The text each option is encoded from, on its own, for scoring."""
        if self.prompt_format == "plain":
            return list(f.option_texts())
        names = f.all_options
        out = []
        for name, text in zip(names, f.option_texts()):
            desc = text[len(name) + 2 :] if text.startswith(f"{name}: ") else ""
            out.append(f'<option name="{name}">{desc}</option>' if desc else f'<option name="{name}"/>')
        return out

    def _question_suffix(self, f, max_option_chars: int = 400) -> str:
        if self.prompt_format == "xml":
            opts = "\n".join(self._option_texts(f))
            instr = f"{self.question_prefix}{f.instructions}" if f.instructions else self.question_prefix.strip()
            text = f'\n<question name="{f.name}">{instr}</question>' if instr else f'\n<question name="{f.name}"/>'
            if len("; ".join(f.option_texts())) <= max_option_chars:  # the same cut-off as the plain format
                text += f"\n<options>\n{opts}\n</options>"
            return text + "\n<answer>"
        opts = "; ".join(f.option_texts())
        text = f"\nquestion: {f.query_text()}"
        if len(opts) <= max_option_chars:
            text += f"\noptions: {opts}"
        return text + "\nanswer:"

    def _pointer_question_block(self, f) -> tuple[list[int], list[int]]:
        """Pointer mode (xml): the question block tokenised piece by piece, so the last token of every option line is
        known exactly. Every option is written out (no length cut-off -- the options ARE the readout). Returns
        (ids, offset of each option's last token within the block)."""
        instr = f"{self.question_prefix}{f.instructions}" if f.instructions else self.question_prefix.strip()
        head = (f'\n<question name="{f.name}">{instr}</question>' if instr else f'\n<question name="{f.name}"/>') + "\n<options>"
        enc = lambda t: self._plain_ids([t], 100_000)[0]  # noqa: E731
        ids, ends, starts = enc(head), [], []
        lines = self._option_texts(f)
        if 0 < len(lines) <= self.option_contrast:
            names = list(f.all_options)
            desc = {n: (t[len(n) + 2:] if t.startswith(f"{n}: ") else "") for n, t in zip(names, f.option_texts())}
            lines = []
            for n in names:
                others = ", ".join(o for o in sorted(names) if o != n)  # labels only, sorted: order cannot leak
                text = f"{desc[n]} (as opposed to: {others})" if desc[n] else f"(as opposed to: {others})"
                lines.append(f'<option name="{n}">{text}</option>')
        for line in lines:
            starts.append(len(ids))
            ids += enc("\n" + line)
            ends.append(len(ids) - 1)
        ids += enc("\n</options>\n<answer>")
        return ids, ends, starts

    def _question_ids(self, schema: Schema) -> list[list[int]]:
        """Token ids of each question block (suffix + tail + pauses), cached per schema."""
        causal = self.pool == "last"
        sep_id = None if causal else getattr(self.tokenizer, "sep_token_id", None)
        tm = schema.task_meta if self.task_meta_pos != "off" else None
        key = ("packed_q", schema, self.task_meta_pos, self.answer_sentinel, self.pause_tokens, self.pause_mode, self.recycle_token,
               self.readout_mode, self.option_contrast)
        if len(self._cache) > 512:  # channel ablation makes every schema distinct
            self._cache.clear()
        if key not in self._cache and self.readout_mode == "letters":
            if self.prompt_format != "xml" or (tm and self.task_meta_pos != "off") or self.pause_tokens or self.recycle_token:
                raise NotImplementedError("letter readout: xml format, no task meta, pause tokens or recycling")
            eos = self.tokenizer.eos_token_id if self.answer_sentinel == "eos" else None
            tail = ([sep_id] if sep_id is not None else []) + ([eos] if eos is not None else [])
            self._cache[key] = [self._letters_question_block(f) + tail for f in schema]
        if key not in self._cache and self.readout_mode == "pointer":
            if self.prompt_format != "xml" or (tm and self.task_meta_pos != "off") or self.pause_tokens or self.recycle_token:
                raise NotImplementedError("pointer readout: xml format, no task meta, pause tokens or recycling")
            plain = lambda f: f.kind == "yesno" and not self.pointer_yesno and self.yesno_layout == "slot"  # noqa: E731
            blocks = [(self._plain_ids([self._question_suffix(f)], 384)[0], None, None) if plain(f)
                      else self._pointer_question_block(f) for f in schema]
            eos = self.tokenizer.eos_token_id if self.answer_sentinel == "eos" else None
            tail = ([sep_id] if sep_id is not None else []) + ([eos] if eos is not None else [])
            self._cache[key] = [ids + tail for ids, _, _ in blocks]
            self._cache[("pointer_ends",) + key] = [ends for _, ends, _ in blocks]
            self._cache[("pointer_starts",) + key] = [starts for _, _, starts in blocks]
        if key not in self._cache:
            if tm and self.task_meta_pos == "stem":  # task sentence folded into each question's own instructions
                from dataclasses import replace as _r
                sent = tm[6:-7] if tm.startswith("<task>") and tm.endswith("</task>") else tm
                suffixes = [self._question_suffix(_r(f, instructions=f"{sent} {f.instructions}".strip())) for f in schema]
            else:
                suffixes = [self._question_suffix(f) for f in schema]
            q = self._plain_ids(suffixes, 192 if self.prompt_format == "plain" else 384)  # tags roughly double the block; never cut the closing answer marker
            eos = self.tokenizer.eos_token_id if self.answer_sentinel == "eos" else None
            tail = ([sep_id] if sep_id is not None else []) + ([eos] if eos is not None else [])
            rec = [self.RECYCLE_ID] if self.recycle_token else []
            self._cache[key] = [ids + tail + self._pause_ids() + rec for ids in q]
        return self._cache[key]

    def _preview_ids(self, schema: Schema) -> list[list[int]]:
        """Question-first previews: each question block as a reader would see it before the text (the question and
        its options, no <answer> marker), then a newline. Cached per schema."""
        key = ("preview", schema, self.question_prefix, self.option_contrast)
        if key not in self._cache:
            txt = []
            for f in schema:
                t = self._question_suffix(f)
                t = t[: -len("\n<answer>")] if t.endswith("\n<answer>") else t.rsplit("\n", 1)[0]
                txt.append(t.lstrip("\n") + "\n")
            self._cache[key] = self._plain_ids(txt, 384)
        return self._cache[key]

    def _state_ids(self, states: list[str], tm: str | None) -> tuple[list[list[int]], int]:
        """Token ids of each state block (tags, task line, cls/sep), and the offset of the state text inside it."""
        causal = self.pool == "last"
        cls_id = None if causal else getattr(self.tokenizer, "cls_token_id", None)
        sep_id = None if causal else getattr(self.tokenizer, "sep_token_id", None)
        s_ids = self._plain_ids(states, max(16, self.max_length - 2))
        state_prefix_extra = 0
        if self.prompt_format == "xml":  # tags added after truncation, so a long state still closes
            open_ids, close_ids = self._plain_ids([f"<{self.state_tag}>\n", f"\n</{self.state_tag}>"], 8)
            pos = self.task_meta_pos
            tmtxt = tm if (tm and pos in ("state_front", "state_end", "block")) else None
            if tmtxt and pos == "state_front":  # task line inside <state>, at the front (the original placement)
                pre = self._plain_ids([f"{tmtxt}\n\n"], 64)[0]
                s_ids = [open_ids + pre + ids + close_ids for ids in s_ids]
                state_prefix_extra = len(open_ids) + len(pre)
            elif tmtxt and pos == "state_end":  # task line inside <state>, at the end (adjacent to the questions)
                suf = self._plain_ids([f"\n\n{tmtxt}"], 64)[0]
                s_ids = [open_ids + ids + suf + close_ids for ids in s_ids]
                state_prefix_extra = len(open_ids)
            elif tmtxt and pos == "block":  # task line as its own block after </state>, before the questions
                blk = self._plain_ids([f"\n{tmtxt}"], 64)[0]
                s_ids = [open_ids + ids + close_ids + blk for ids in s_ids]
                state_prefix_extra = len(open_ids)
            else:
                s_ids = [open_ids + ids + close_ids for ids in s_ids]
                state_prefix_extra = len(open_ids)
        if self.state_preamble:
            pre = self._plain_ids([f"{self.state_preamble}\n"], 64)[0]
            s_ids = [pre + ids for ids in s_ids]
            state_prefix_extra += len(pre)
        s_ids = [([cls_id] if cls_id is not None else []) + ids + ([sep_id] if sep_id is not None else []) for ids in s_ids]
        return s_ids, state_prefix_extra

    def _packed_slots(self, states: list[str], schema: Schema, exclude: list[list[str]] | None = None) -> torch.Tensor:
        causal = self.pool == "last"
        tok = self.tokenizer
        cls_id = None if causal else getattr(tok, "cls_token_id", None)
        tm = schema.task_meta if self.task_meta_pos != "off" else None
        q_ids = self._question_ids(schema)
        q_total = sum(len(x) for x in q_ids)
        s_ids, state_prefix_extra = self._state_ids(states, tm)
        if self.question_first and self.packed_impl != "replicate":
            raise NotImplementedError("question-first reading is implemented for packed_impl='replicate' only")
        if self.packed_impl == "replicate":
            if exclude is not None:
                raise NotImplementedError("feature exclusion is implemented for packed_impl='mask' only")
            if self.readout_mode == "pointer":
                raise NotImplementedError("pointer readout is implemented for packed_impl='mask' only (so far)")
            return self._replicated_slots(s_ids, q_ids, causal, self._preview_ids(schema) if self.question_first else None)

        batch = len(states)
        n = max(len(x) for x in s_ids) + q_total
        pad = getattr(tok, "pad_token_id", None) or 0
        ids = torch.full((batch, n), pad, dtype=torch.long)
        pos = torch.zeros((batch, n), dtype=torch.long)
        allow = torch.eye(n, dtype=torch.bool).unsqueeze(0).repeat(batch, 1, 1)  # pads attend to themselves only
        spans: list[list[tuple[int, int]]] = []
        # Feature exclusion: mask the given substrings of each state so NO query
        # (state token, option, or question) attends to them, throughout the backbone.
        # The excluded content is causally inert -- stronger and artefact-free versus a
        # text placeholder (which is itself a token that carries signal).
        excluded_cols = None
        if exclude is not None:
            ml = max(16, self.max_length - 2)
            prefix = (1 if cls_id is not None else 0) + state_prefix_extra
            excluded_cols = []
            for text, subs in zip(states, exclude):
                enc = self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=ml, return_offsets_mapping=True)
                chars = set()
                for sub in subs or []:
                    i = text.find(sub)
                    while i != -1:
                        chars.update(range(i, i + len(sub)))
                        i = text.find(sub, i + 1)
                cols = {prefix + t for t, (a, e) in enumerate(enc["offset_mapping"]) if a != e and not chars.isdisjoint(range(a, e))}
                excluded_cols.append(cols)
        for b, st in enumerate(s_ids):
            S = len(st)
            ids[b, :S] = torch.tensor(st)
            # When state tokens are excluded, collapse their positions so the layout the
            # model reads is invariant to the excluded span's token length (otherwise a
            # 1-token vs 2-token name shifts every downstream position). Excluded tokens
            # are inert (masked below), so parking them on their neighbour's position is
            # harmless; what matters is that non-excluded tokens and the questions get
            # positions independent of what was excluded.
            Eb = {c for c in excluded_cols[b] if c < S} if excluded_cols is not None else set()
            if Eb:
                posrow, pnext = [], 0
                for i in range(S):
                    posrow.append(pnext)
                    if i not in Eb:
                        pnext += 1
                pos[b, :S] = torch.tensor(posrow)
                s_eff = pnext
            else:
                pos[b, :S] = torch.arange(S)
                s_eff = S
            block = torch.ones(S, S, dtype=torch.bool)
            state_causal = causal and self.packed_state_attention == "native"
            allow[b, :S, :S] = torch.tril(block) if state_causal else block
            start, row = S, []
            for qn, qi in enumerate(q_ids):
                L = len(qi)
                ids[b, start : start + L] = torch.tensor(qi)
                pos[b, start : start + L] = torch.arange(s_eff, s_eff + L)  # questions start after the non-excluded state
                allow[b, start : start + L, :S] = True
                qb = torch.ones(L, L, dtype=torch.bool)
                allow[b, start : start + L, start : start + L] = torch.tril(qb) if causal else qb
                if self.readout_mode == "pointer" and self.option_isolation:
                    self._isolate_options(allow[b], pos[b], start, L, s_eff, qn, schema)
                row.append((start, start + L))
                start += L
            spans.append(row)
        if excluded_cols is not None:
            for b, cols in enumerate(excluded_cols):
                for c in cols:
                    if c < n:
                        allow[b, :, c] = False  # no query attends to the excluded key
                        allow[b, c, c] = True   # keep its own row valid (no empty-softmax NaN)
        ids, pos, allow = ids.to(self.device), pos.to(self.device), allow.to(self.device)

        # Rule: compose the block mask with the backbone's native mask, never
        # substitute it. A custom mask replaces whatever structure the model builds
        # for itself (sliding windows, local/global alternation), and the result
        # runs without error while no longer being the pretrained model.
        config = self.encoder.config
        window = getattr(config, "local_attention", None)
        layer_types = set(getattr(config, "layer_types", None) or [])
        structured = "sliding_attention" in layer_types or bool(getattr(config, "use_sliding_window", False))
        if structured and getattr(config, "model_type", "") != "modernbert":
            raise NotImplementedError(
                f"{config.model_type} uses structured (sliding-window) attention; packed mode must compose "
                "its block mask with that structure before this backbone can be used"
            )
        if getattr(config, "model_type", "") == "modernbert" and window:
            # ModernBERT's local layers were pretrained with a sliding window; a custom
            # mask replaces the model's own, so the window has to be rebuilt here, on
            # position ids (questions restart after the state, so they sit by its tail).
            near = (pos.unsqueeze(2) - pos.unsqueeze(1)).abs() <= window // 2
            mask = {"full_attention": self._additive(allow), "sliding_attention": self._additive(allow & near)}
        else:
            mask = self._additive(allow)
        if self.lora_scope == "questions":
            tm = torch.zeros(ids.shape, device=ids.device)
            for b, row in enumerate(spans):
                for a, e in row:
                    tm[b, a:e] = 1.0
            self._tok_mask = tm
        try:
            hidden = self._backbone_hidden(**self._backbone_inputs(ids), attention_mask=mask, position_ids=pos)
        finally:
            self._tok_mask = None if not torch.is_grad_enabled() else self._tok_mask

        slots = []
        if self.span_pool == "attn":
            mode = "attn"
        elif (not causal) or self.span_pool == "mean":
            mode = "mean"
        else:
            mode = "last"
        for b in range(batch):
            if mode == "attn":  # learned attention-pool over each span (hidden is fp32 here)
                row = []
                for s_, e in spans[b]:
                    sl = hidden[b, s_:e]  # (L, d)
                    w = torch.softmax(self.span_attn(sl).squeeze(-1), dim=0)  # (L,)
                    row.append((w.unsqueeze(-1) * sl).sum(0))
            else:
                row = [hidden[b, s_:e].mean(0) if mode == "mean" else hidden[b, e - 1] for s_, e in spans[b]]
            slots.append(torch.stack(row))
        if self.readout_mode == "pointer":  # each option's in-context vector: (B, k_f, d) per field
            key = ("packed_q", schema, self.task_meta_pos, self.answer_sentinel, self.pause_tokens, self.pause_mode,
                   self.recycle_token, self.readout_mode, self.option_contrast)
            ends = self._cache[("pointer_ends",) + key]
            self._pointer_opts = [None if ends[i] is None else
                                  torch.stack([hidden[b, [spans[b][i][0] + o for o in ends[i]]] for b in range(batch)])
                                  for i in range(len(ends))]
        return torch.stack(slots)  # (B, F, d)

    def _isolate_options(self, allow_b, pos_b, start, L, s_eff, qn, schema) -> None:
        """Option isolation inside one question block (in place on one row's mask and positions): option lines see
        the state, the question head and themselves only; all start at the same position; the tail (</options>,
        <answer>, any sentinel) sees head + every option + itself, positioned after the longest option."""
        key = ("packed_q", schema, self.task_meta_pos, self.answer_sentinel, self.pause_tokens, self.pause_mode,
               self.recycle_token, self.readout_mode, self.option_contrast)
        starts = self._cache[("pointer_starts",) + key][qn]
        ends = self._cache[("pointer_ends",) + key][qn]
        if starts is None:  # a question rendered in the plain layout: nothing to isolate
            return
        h = starts[0]
        spans = [(a, e + 1) for a, e in zip(starts, ends)]
        longest = max(e - a for a, e in spans)
        t0 = spans[-1][1]
        blk = allow_b[start : start + L, start : start + L]
        for a, e in spans:  # an option line: head + itself (causal), no other option
            blk[a:e, h:t0] = False
            blk[a:e, a:e] = torch.tril(torch.ones(e - a, e - a, dtype=torch.bool))
            pos_b[start + a : start + e] = torch.arange(s_eff + h, s_eff + h + (e - a))
        pos_b[start + t0 : start + L] = torch.arange(s_eff + h + longest, s_eff + h + longest + (L - t0))

    def _replicated_slots(self, s_ids: list[list[int]], q_ids: list[list[int]], causal: bool,
                          pre_ids: list[list[int]] | None = None) -> torch.Tensor:
        """Packed semantics without a custom mask: one right-padded row [state | q_i] per (state, question),
        run under the backbone's native causal attention. Positions are the natural 0..n, i.e. the same as
        the mask path (questions start where the state ends). Returns (B, F, d)."""
        rows = self._replicated_rows(s_ids, [q_ids] * len(s_ids), causal, None if pre_ids is None else [pre_ids] * len(s_ids))
        return torch.stack(rows)

    def _replicated_rows(self, s_ids: list[list[int]], q_per_state: list[list[list[int]]], causal: bool,
                         pre_per_state: list[list[list[int]]] | None = None) -> list[torch.Tensor]:
        """As _replicated_slots, but every state brings its OWN question blocks (heterogeneous batches).
        `pre_per_state` (question-first reading): a block per question placed before the state in that question's
        row, [pre_i | state | q_i]. Returns one (F_b, d) tensor per state."""
        if not causal:
            raise NotImplementedError("packed_impl='replicate' needs a causal backbone")
        tok = self.tokenizer
        pad = getattr(tok, "pad_token_id", None) or 0
        if pre_per_state is None:
            pre_per_state = [[[] for _ in qs] for qs in q_per_state]
        trip = [(pi, st, qi) for st, qs, ps in zip(s_ids, q_per_state, pre_per_state) for qi, pi in zip(qs, ps)]
        rows = [pi + st + qi for pi, st, qi in trip]
        ends = [len(pi) + len(st) + len(qi) for pi, st, qi in trip]
        starts = [len(pi) + len(st) for pi, st, _ in trip]
        n = max(len(r) for r in rows)
        ids = torch.full((len(rows), n), pad, dtype=torch.long)
        attn = torch.zeros((len(rows), n), dtype=torch.long)
        for j, r in enumerate(rows):
            ids[j, : len(r)] = torch.tensor(r)
            attn[j, : len(r)] = 1
        ids, attn = ids.to(self.device), attn.to(self.device)
        pos = torch.arange(n, device=self.device).unsqueeze(0).expand(len(rows), n)
        if self.lora_scope == "questions":
            tm = torch.zeros((len(rows), n), device=self.device)
            for j, (a, e) in enumerate(zip(starts, ends)):
                tm[j, a:e] = 1.0
            self._tok_mask = tm
        try:
            hidden = self._backbone_hidden(**self._backbone_inputs(ids), attention_mask=attn, position_ids=pos)
        finally:
            self._tok_mask = None if not torch.is_grad_enabled() else self._tok_mask
        mode = "attn" if self.span_pool == "attn" else ("mean" if self.span_pool == "mean" else "last")
        out = []
        for j, (a, e) in enumerate(zip(starts, ends)):
            if mode == "attn":
                sl = hidden[j, a:e]
                w = torch.softmax(self.span_attn(sl).squeeze(-1), dim=0)
                out.append((w.unsqueeze(-1) * sl).sum(0))
            else:
                out.append(hidden[j, a:e].mean(0) if mode == "mean" else hidden[j, e - 1])
        out = torch.stack(out)
        per, i = [], 0
        for qs in q_per_state:
            per.append(out[i:i + len(qs)])
            i += len(qs)
        return per

    def _read(self, queries: torch.Tensor, hidden: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        """Field queries (B, Q, d) read the state tokens (B, L, d) -> slots (B, Q, d)."""
        if self.readout_stack is None:
            slots, _ = self.readout(queries, hidden, hidden, key_padding_mask=key_padding)
            return self.readout_norm(slots + queries)  # residual, then norm
        slots = queries
        for layer in self.readout_stack:
            slots = layer(slots, hidden, key_padding)
        return slots

    @staticmethod
    def _question_prefix(f, max_option_chars: int = 400) -> str:
        """The question as text, placed before the state. Options are listed when
        they fit; a 100-option field would otherwise crowd the state out."""
        opts = "; ".join(f.option_texts())
        lines = [f"question: {f.query_text()}"]
        if len(opts) <= max_option_chars:
            lines.append(f"options: {opts}")
        return "\n".join(lines) + "\n\n"

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- forward -----------------------------------------------------------

    def forward(self, states: list[str], schema: Schema, exclude: list[list[str]] | None = None) -> list[torch.Tensor]:
        """States -> a list (length F) of per-field logit tensors, each (B, k_f)."""
        if exclude is not None and self.question_mode != "packed":
            raise NotImplementedError("feature exclusion is implemented for packed question mode only")
        if (self.readout_mode == "pointer" and self.pointer_yesno and not self.recycle_token) or self.readout_mode == "letters":
            # every field is scored from its in-context option states: the separate option encodings (a full extra
            # backbone pass over all option texts per step) would be computed and never used
            field_q, option_embs = None, [None] * len(schema)
        else:
            field_q, option_embs = self._schema_embeddings(schema, cache_ok=self._encoder_frozen)
        if self.recycle_token and self.question_mode == "packed":
            import random as _random
            from contextlib import nullcontext
            rounds = _random.randint(0, self.recycle_rounds) if self.training else self.eval_rounds
            probs = [torch.full((len(states), o.size(0)), 1.0 / o.size(0), device=o.device) for o in option_embs]
            history = []
            try:
                for r in range(rounds + 1):
                    graded = r == rounds
                    with (nullcontext() if graded else torch.no_grad()):
                        self._recycle_vecs = self._estimate_vectors(probs, option_embs if graded else [o.detach() for o in option_embs])
                        logits = self._score(states, schema, field_q, option_embs, exclude)
                    probs = [F.softmax(lg.detach().float(), dim=-1) for lg in logits]
                    history.append(probs)
            finally:
                self._recycle_vecs = None
            self.last_round_probs = history
            return logits
        return self._score(states, schema, field_q, option_embs, exclude)

    def _score(self, states, schema, field_q, option_embs, exclude=None) -> list[torch.Tensor]:
        """One pass: slots for every (state, question), scored against the options. -> per-field (B, k_f) logits."""
        batch = len(states)
        n_fields = len(schema)

        if self.question_mode == "packed":
            slots = self._packed_slots(states, schema, exclude=exclude)  # (B, F, d), one pass, block mask
        elif not self.question_in_state:
            hidden, attn = self._encode_texts(states)  # (B, L, d), one pass for every field
            queries = field_q.unsqueeze(0).expand(batch, n_fields, -1)  # (B, F, d)
            slots = self._read(queries, hidden, attn == 0)
        else:
            per_field = []
            for i, f in enumerate(schema):
                prefix = self._question_prefix(f)
                hidden, attn = self._encode_texts([prefix + st for st in states], bidirectional=True)  # one pass per field
                q = field_q[i].view(1, 1, -1).expand(batch, 1, -1)
                per_field.append(self._read(q, hidden, attn == 0))
            slots = torch.cat(per_field, dim=1)  # (B, F, d)
        slots_p = self.proj(slots)  # (B, F, proj)

        temp = self.temperature()
        logits: list[torch.Tensor] = []
        if self.readout_mode == "letters":  # the <answer> state against the code rows of the (tied) LM head
            for i, f in enumerate(schema):
                if f.n_outputs != len(f.options):
                    raise NotImplementedError("letter readout: options outside the listed ones (e.g. none_of_these)")
                logits.append(slots[:, i, :].float() @ self.code_readout[: len(f.options)].t())
            return logits
        if self.readout_mode == "pointer":  # bilinear: the <answer> vector against each option's in-context vector
            opts, self._pointer_opts = self._pointer_opts, None
            for i, f in enumerate(schema):
                if f.kind == "yesno" and not self.pointer_yesno:  # hybrid: the slot readout for yes/no
                    opt_p, slot = self.proj(option_embs[i]), slots_p[:, i, :]
                    if self.normalize:
                        slot, opt_p = F.normalize(slot, dim=-1), F.normalize(opt_p, dim=-1)
                    logits.append((slot @ opt_p.t()) / temp)
                    continue
                if f.n_outputs != len(f.options):
                    raise NotImplementedError("pointer readout: options outside the listed ones (e.g. none_of_these)")
                q = self.pointer_q(slots[:, i, :].float())  # (B, dp)
                k = self.pointer_k(opts[i].float())  # (B, k_f, dp)
                logits.append(torch.einsum("bd,bkd->bk", q, k) * self._pointer_scale)
            return logits
        for i in range(n_fields):
            opt_p = self.proj(option_embs[i])  # (k_f, proj)
            slot = slots_p[:, i, :]
            if self.option_block is not None:
                # per example, since what the options make of each other depends on the state
                x = torch.cat([slot.unsqueeze(1), opt_p.unsqueeze(0).expand(batch, -1, -1)], dim=1)  # (B, 1 + k, proj)
                for layer in self.option_block:
                    x = layer(x)
                slot, opts = x[:, 0], x[:, 1:]
                if self.normalize:
                    slot, opts = F.normalize(slot, dim=-1), F.normalize(opts, dim=-1)
                score = torch.einsum("bd,bkd->bk", slot, opts)
            else:
                if self.normalize:
                    slot, opt_p = F.normalize(slot, dim=-1), F.normalize(opt_p, dim=-1)
                score = slot @ opt_p.t()  # (B, k_f)
            logits.append(score / temp)
        return logits

    @torch.no_grad()
    def predict(self, states: list[str], schema: Schema, exclude: list[list[str]] | None = None) -> list[dict[str, list[float]]]:
        """Convenience: states -> list of {field_name: [probabilities]} per example."""
        was_training = self.training
        self.eval()
        logits = self.forward(states, schema, exclude=exclude)
        probs = [F.softmax(l, dim=-1) for l in logits]
        out = []
        for b in range(len(states)):
            out.append(
                {schema.fields[i].name: probs[i][b].cpu().tolist() for i in range(len(schema))}
            )
        if was_training:
            self.train()
        return out

    def decide(self, states: list, schema: Schema, exclude: list[list[str]] | None = None) -> list[dict[str, Answer]]:
        """States (str, dict or list each) -> typed answers per field, one dict per state.

        This is the API surface: every question in ``schema`` is answered in the
        same forward pass, independently, with a full distribution, a confidence,
        and the kind-specific reading (choice / score / yesno)."""
        import warnings

        features = getattr(self, "trained_features", set())
        reads_stems = "channel_ablation" in features
        bare = [f.name for f in schema if f.kind == "yesno" and not (f.descriptions and any(f.descriptions)) and not (reads_stems and f.instructions)]
        if bare:
            warnings.warn(
                f"yes/no field(s) {bare} carry nothing this checkpoint reads. Every yes/no question shares the option "
                "names 'no' and 'yes'; checkpoints trained without channel ablation tell them apart only by their option "
                "descriptions and ignore the instructions, so the answers are unreliable and may invert. "
                "Pass criteria={'no': ..., 'yes': ...}, or use a checkpoint trained with channel ablation and give instructions.",
                stacklevel=2,
            )
        unseen_none = [f.name for f in schema if f.none_option] if "none_option" not in features else []
        if unseen_none:
            warnings.warn(
                f"field(s) {unseen_none} offer none_of_these, but this checkpoint was not trained with the none option. "
                "To it that is an untrained option: it will rarely be chosen even when nothing offered fits. "
                "Remove none_option, or use a checkpoint trained with --none-toggle / --none-option.",
                stacklevel=2,
            )
        preds = self.predict([render_state(s) for s in states], schema, exclude=exclude)
        return [answers_for(schema, p) for p in preds]


CAUSAL_MODEL_TYPES = {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "gemma3", "gemma3_text", "phi3", "olmo2", "smollm3",
                      "qwen3_5", "qwen3_5_text", "granite", "granitemoehybrid", "olmo3", "ministral3", "mistral3", "nemotron_h"}
# backbones whose layers cannot take a custom attention mask (recurrent / linear attention): packed mode replicates
HYBRID_MODEL_TYPES = {"qwen3_5", "qwen3_5_text", "granitemoehybrid", "nemotron_h"}
def _enable_hub_conv1d() -> None:
    """Qwen3.5's linear-attention layers fall back to a slow PyTorch conv unless a package named causal_conv1d
    imports. coxlm/_shims/causal_conv1d forwards to the Hugging Face hub build (no compilation); matches the reference within
    bf16 (hidden rel 0.9%, grad rel 1.8%), ~1.1x faster on a 2B. Off with COX_HUB_CONV1D=0 or without `kernels`."""
    import importlib.util
    shims = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shims")
    if os.environ.get("COX_HUB_CONV1D", "1") != "0" and importlib.util.find_spec("kernels") and shims not in sys.path:
        sys.path.insert(0, shims)


MAMBA_MODEL_TYPES = {"granitemoehybrid", "nemotron_h"}
# multimodal checkpoints: the text model is the backbone (the vision tower is dropped)
MULTIMODAL_TEXT = {"qwen3_5": "language_model", "mistral3": "language_model"}


def build_model(
    encoder_name: str = "answerdotai/ModernBERT-base",
    freeze_encoder: bool = True,
    dtype: str | None = None,
    pool: str | None = None,
    state_norm: bool | str | None = None,
    lora_r: int = 0,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    grad_checkpoint: bool = False,
    truncate: bool = False,
    **kwargs,
) -> AmortizedDecisionModel:
    """Build the model on a pretrained backbone (downloads on first use).

    Any model that returns per-token hidden states works: a bidirectional
    encoder (ModernBERT, DeBERTa) or a decoder LLM used as an encoder (Qwen3
    base). ``dtype`` "bf16" loads the backbone in bfloat16 (the head stays
    fp32); ``pool`` defaults to "last" for causal model types and "mean"
    otherwise.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(encoder_name)
    causal = config.model_type in CAUSAL_MODEL_TYPES or getattr(config, "is_decoder", False)
    if pool is None:
        pool = "last" if causal else "mean"
    if state_norm is None:
        state_norm = "standardize" if causal else "off"
    torch_dtype = {None: None, "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kw = {"dtype": torch_dtype} if torch_dtype else {}
    if config.model_type in HYBRID_MODEL_TYPES and config.model_type not in MAMBA_MODEL_TYPES:
        _enable_hub_conv1d()
    if config.model_type in MAMBA_MODEL_TYPES:
        try:  # prebuilt Hub kernels (kernels-community/mamba-ssm): the reference Mamba scan is too slow / memory-hungry
            import kernels  # noqa: F401
            load_kw["use_kernels"] = True
        except ImportError:
            pass
    encoder = AutoModel.from_pretrained(encoder_name, **load_kw)
    if config.model_type in MULTIMODAL_TEXT:
        # load the whole checkpoint (so every weight name matches), then keep only the text model
        encoder = getattr(encoder, MULTIMODAL_TEXT[config.model_type])
    hidden = encoder.config.hidden_size
    if grad_checkpoint:
        encoder.gradient_checkpointing_enable()
        if hasattr(encoder, "enable_input_require_grads"):
            encoder.enable_input_require_grads()
    model = AmortizedDecisionModel(
        encoder, tokenizer, hidden_size=hidden, freeze_encoder=freeze_encoder or lora_r > 0, pool=pool, state_norm=state_norm, **kwargs
    )
    if truncate:
        model.truncate_backbone()  # before LoRA, so no adapters are made for dropped blocks
    if lora_r > 0:
        # Low-rank adapters on every linear layer of the (frozen) backbone: the
        # way to fine-tune a multi-billion-parameter LLM as an encoder. The base
        # weights stay frozen and in their loaded dtype; only the adapters and
        # the readout head train. Schema embeddings must be recomputed each step
        # so the adapters receive gradient through them, hence _encoder_frozen.
        from peft import LoraConfig, get_peft_model

        # Mamba mixers: PEFT refuses LoRA on their out_proj / conv1d; the other projections and attention/MLP still adapt
        exclude = ["out_proj", "conv1d"] if config.model_type in MAMBA_MODEL_TYPES else None
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha or 2 * lora_r, lora_dropout=lora_dropout, target_modules="all-linear",
                         exclude_modules=exclude)
        model.encoder = get_peft_model(model.encoder, cfg)
        for n_, p_ in model.encoder.named_parameters():
            if "lora_" in n_:
                p_.data = p_.data.float()  # adapters in fp32 even on a bf16 base
        model._encoder_frozen = False
        model.install_lora_scope()
    return model
