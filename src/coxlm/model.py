"""The decision model: a pretrained backbone read once per question, one forward pass for a whole question set.

The state (the text being judged) and each question are written as XML-tagged text. Every question block is read
after the state: each question sees the state and itself, never another question, and the state never sees a
question. Each question's answer vector is the hidden state at the end of its block (``<answer>``). Two readouts
turn that vector into a distribution over the question's options:

  "slot"     cosine similarity between the projected answer vector and each option encoded on its own, divided by a
             learned temperature.
  "pointer"  a bilinear score between the answer vector and each option's hidden state in context (the last token of
             that option's line in the question block). With option isolation each option line sees the state, the
             question and itself only, and all option lines start at the same position, so the order the options are
             listed in cannot change the answer.

Questions are kept apart in one of two ways. "mask": one sequence [state | q1 | q2 | ...] under a block attention
mask, positions restarting after the state for every question. "replicate": one row [state | q_i] per question under
the backbone's own causal mask, for backbones whose recurrent or linear-attention layers cannot take a custom mask
(Qwen3.5 and other hybrids). Both give every question exactly the same view.

The pointer readout on a hybrid backbone uses cache forks instead: the states are run once with the cache on, and
the cache is then copied per branch row. Each option line continues from the state and its question head alone (one
branch per option), the <answer> vector from the head and the block's closing tags alone (it does not see the options:
separate branches' recurrent states cannot be merged), and a yes/no question scored by the slot readout from the
state with its whole block. This is the layout the research models were trained with.

coxlm ships this module for inference only.
"""
from __future__ import annotations

import copy
import os
import sys
from typing import Any, Iterable, Iterator, overload

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decide import Answers, Q, answers_for, as_schema, iter_batches, render_state
from .schema import Schema


class AmortizedDecisionModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        tokenizer,
        hidden_size: int,
        proj_dim: int | None = None,
        max_length: int = 256,
        normalize: bool = True,
        init_temp: float | None = None,
        pool: str = "mean",
        state_norm: bool | str = False,
        packed_impl: str = "auto",
    ) -> None:
        """``pool`` is how an option text encoded on its own is collapsed into one vector: "mean" over tokens
        (bidirectional encoders) or the "last" token (causal decoders, where only the final position has seen the
        whole text). It also decides how a question block is read: its mean (bidirectional) or its last token."""
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.normalize = normalize
        if pool not in ("mean", "last"):
            raise ValueError("pool must be 'mean' or 'last'")
        self.pool = pool
        # Encoder hidden states carry a few "massive activation" dimensions that hold most of the energy, so cosine
        # scores see mostly those dimensions. Standardising each dimension with fixed statistics (calibrated once,
        # stored in the checkpoint) removes them.
        mode = {False: "off", None: "off"}.get(state_norm, state_norm)
        if mode not in ("off", "standardize"):
            raise ValueError("state_norm must be off / standardize")
        self.state_norm_mode = mode
        if mode == "standardize":
            self.register_buffer("state_mu", torch.zeros(hidden_size))
            self.register_buffer("state_sd", torch.ones(hidden_size))
            self.register_buffer("state_calibrated", torch.zeros((), dtype=torch.bool))
        if packed_impl not in ("auto", "mask", "replicate"):
            raise ValueError("packed_impl must be auto / mask / replicate")
        if packed_impl == "auto":
            lt = set(getattr(encoder.config, "layer_types", None) or [])
            hybrid = bool(lt & {"linear_attention", "mamba", "recurrent"}) or getattr(encoder.config, "model_type", "") in HYBRID_MODEL_TYPES
            packed_impl = "replicate" if hybrid else "mask"
        self.packed_impl = packed_impl
        self.readout_mode = "slot"  # or "pointer" (set_readout)
        self.pointer_q = self.pointer_k = None
        self.option_isolation = False  # pointer readout: options never see each other (order-free)
        self.pointer_yesno = True  # pointer readout: False scores yes/no questions with the slot readout instead
        self.proj = nn.Linear(hidden_size, proj_dim or hidden_size, bias=False)
        # Scores are cosine similarities when ``normalize`` is on, so they live in [-1, 1] and the temperature sets
        # the logit scale directly (CLIP-style). temperature = softplus(log_temp) + eps.
        if init_temp is None:
            init_temp = 0.07 if normalize else 1.0
        self.log_temp = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(init_temp)))))
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._cache: dict = {}

    @property
    def device(self) -> torch.device:
        return self.proj.weight.device

    def close(self) -> None:
        """Release the model: drops the backbone and readout weights and the cached option encodings, and returns the
        freed GPU memory to the device. Later calls raise RuntimeError. ``with coxlm.load(...) as model:`` calls
        this at the end of the block."""
        import gc

        if getattr(self, "closed", False):
            return
        dev = self.device
        self._cache.clear()
        self.__dict__.pop("_tok_memo", None)
        self.encoder = None
        self.pointer_q = self.pointer_k = None
        self._modules.clear()
        self._parameters.clear()
        self._buffers.clear()
        self.closed = True
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self) -> "AmortizedDecisionModel":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def set_readout(self, mode: str, dp: int = 256) -> None:
        """'slot' (default) or 'pointer' (see the module docstring). Pointer mode adds a bilinear head (dp = its width)."""
        if mode not in ("slot", "pointer"):
            raise ValueError(f"unknown readout mode {mode!r}")
        self.readout_mode = mode
        self.pointer_q = self.pointer_k = None
        if mode == "pointer":
            d = self.encoder.get_input_embeddings().weight.shape[1]
            dev = self.encoder.get_input_embeddings().weight.device
            self.pointer_q, self.pointer_k = nn.Linear(d, dp).to(dev), nn.Linear(d, dp).to(dev)
            self._pointer_scale = dp ** -0.5
        self._cache.clear()

    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temp) + 1e-4

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- encoding ------------------------------------------------------------

    def _backbone_hidden(self, **kw) -> torch.Tensor:
        """Run the backbone; the last hidden state in the head's dtype, standardised if the model is set to."""
        return self._head_view(self.encoder(**kw))

    def _head_view(self, out) -> torch.Tensor:
        """A backbone output as the head sees it: the last hidden state in the head's dtype, standardised if set."""
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hidden = hidden.to(self.proj.weight.dtype)  # the backbone may run in bf16; the head stays fp32
        if self.state_norm_mode == "standardize":
            hidden = (hidden - self.state_mu) / self.state_sd
        return hidden

    def _encode_texts(self, texts: list[str]):
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        return self._backbone_hidden(input_ids=input_ids, attention_mask=attn), attn

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        mask = attn.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    @staticmethod
    def _last_pool(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """Hidden state at each sequence's last non-pad token (works for either padding side)."""
        if bool((attn[:, -1] == 1).all()):  # left padding: last column is always real
            return hidden[:, -1, :]
        idx = attn.sum(1).clamp(min=1) - 1  # right padding
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx, :]

    def _pool(self, hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        return self._last_pool(hidden, attn) if self.pool == "last" else self._mean_pool(hidden, attn)

    def _option_embeddings(self, schema: Schema) -> list[torch.Tensor]:
        """Each field's options, each encoded on its own: [(k_f, d)] per field, cached per schema."""
        # keyed on the schema's content (it is frozen and hashable), not its object id: ids are recycled after
        # garbage collection, and a stale hit returns another schema's embeddings
        if len(self._cache) > 512:
            self._cache.clear()
        key = ("options", schema)
        if key not in self._cache:
            out = []
            for f in schema:
                oh, oa = self._encode_texts(self._option_texts(f))
                out.append(self._pool(oh, oa).detach())  # (k_f, d)
            self._cache[key] = out
        return self._cache[key]

    def _additive(self, allow: torch.Tensor) -> torch.Tensor:
        """(B, n, n) bool 'may attend' -> (B, 1, n, n) additive mask in the encoder's dtype."""
        dtype = next(self.encoder.parameters()).dtype
        mask = torch.zeros(allow.shape, dtype=dtype, device=allow.device)
        mask.masked_fill_(~allow, torch.finfo(dtype).min)
        return mask.unsqueeze(1)

    def _plain_ids(self, texts: list[str], max_length: int) -> list[list[int]]:
        """Token ids per text (no special tokens), memoised. Returns fresh lists (callers extend them)."""
        cache = self.__dict__.setdefault("_tok_memo", {})
        if len(cache) > 200_000:
            cache.clear()
        missing = list(dict.fromkeys(t for t in texts if (t, max_length) not in cache))
        if missing:
            enc = self.tokenizer(missing, add_special_tokens=False, truncation=True, max_length=max_length, padding=False, return_tensors=None)
            for t, ids in zip(missing, enc["input_ids"]):
                cache[(t, max_length)] = tuple(ids)
        return [list(cache[(t, max_length)]) for t in texts]

    # -- prompt layout -------------------------------------------------------

    def _option_texts(self, f) -> list[str]:
        """Each option as an XML line: <option name="...">description</option>."""
        out = []
        for name, text in zip(f.all_options, f.option_texts()):
            desc = text[len(name) + 2 :] if text.startswith(f"{name}: ") else ""
            out.append(f'<option name="{name}">{desc}</option>' if desc else f'<option name="{name}"/>')
        return out

    @staticmethod
    def _question_head(f) -> str:
        return f'\n<question name="{f.name}">{f.instructions}</question>' if f.instructions else f'\n<question name="{f.name}"/>'

    def _question_suffix(self, f, max_option_chars: int = 400) -> str:
        """The slot readout's question block. Options are listed when they fit; a 100-option field would otherwise
        crowd the state out (each option is also encoded on its own for scoring)."""
        text = self._question_head(f)
        if len("; ".join(f.option_texts())) <= max_option_chars:
            text += "\n<options>\n" + "\n".join(self._option_texts(f)) + "\n</options>"
        return text + "\n<answer>"

    def _pointer_question_block(self, f) -> tuple[list[int], list[int], list[int]]:
        """The pointer readout's question block, tokenised piece by piece so the first and last token of every
        option line are known exactly. Every option is written out (the options ARE the readout).
        Returns (ids, offset of each option's last token, offset of each option's first token)."""
        enc = lambda t: self._plain_ids([t], 100_000)[0]  # noqa: E731
        ids, ends, starts = enc(self._question_head(f) + "\n<options>"), [], []
        for line in self._option_texts(f):
            starts.append(len(ids))
            ids += enc("\n" + line)
            ends.append(len(ids) - 1)
        ids += enc("\n</options>\n<answer>")
        return ids, ends, starts

    def _question_ids(self, schema: Schema) -> list[list[int]]:
        """Token ids of each question block, cached per schema. In pointer mode, also caches each block's option
        line offsets (None for a yes/no question scored by the slot readout)."""
        key = ("questions", schema, self.readout_mode, self.pointer_yesno)
        if len(self._cache) > 512:
            self._cache.clear()
        if key not in self._cache:
            sep_id = None if self.pool == "last" else getattr(self.tokenizer, "sep_token_id", None)
            tail = [sep_id] if sep_id is not None else []
            if self.readout_mode == "pointer":
                plain = lambda f: f.kind == "yesno" and not self.pointer_yesno  # noqa: E731
                blocks = [(self._plain_ids([self._question_suffix(f)], 384)[0], None, None) if plain(f)
                          else self._pointer_question_block(f) for f in schema]
                self._cache[key] = [ids + tail for ids, _, _ in blocks]
                self._cache[("ends",) + key] = [ends for _, ends, _ in blocks]
                self._cache[("starts",) + key] = [starts for _, _, starts in blocks]
            else:
                q = self._plain_ids([self._question_suffix(f) for f in schema], 384)  # never cut the closing <answer>
                self._cache[key] = [ids + tail for ids in q]
        return self._cache[key]

    def _state_ids(self, states: list[str]) -> list[list[int]]:
        """Token ids of each state block: <state> ... </state> (tags added after truncation, so a long state still
        closes), wrapped in cls / sep for bidirectional encoders."""
        causal = self.pool == "last"
        cls_id = None if causal else getattr(self.tokenizer, "cls_token_id", None)
        sep_id = None if causal else getattr(self.tokenizer, "sep_token_id", None)
        open_ids, close_ids = self._plain_ids(["<state>\n", "\n</state>"], 8)
        s_ids = [open_ids + ids + close_ids for ids in self._plain_ids(states, max(16, self.max_length - 2))]
        return [([cls_id] if cls_id is not None else []) + ids + ([sep_id] if sep_id is not None else []) for ids in s_ids]

    # -- one pass over states and questions -----------------------------------

    def _packed_slots(self, states: list[str], schema: Schema) -> torch.Tensor:
        """Every question's answer vector for every state: (B, F, d)."""
        causal = self.pool == "last"
        q_ids = self._question_ids(schema)
        s_ids = self._state_ids(states)
        if self.packed_impl == "replicate":
            if self.readout_mode == "pointer":
                return self._forked_pointer_slots(s_ids, q_ids, schema)
            return self._replicated_slots(s_ids, q_ids)

        batch = len(states)
        n = max(len(x) for x in s_ids) + sum(len(x) for x in q_ids)
        pad = getattr(self.tokenizer, "pad_token_id", None) or 0
        ids = torch.full((batch, n), pad, dtype=torch.long)
        pos = torch.zeros((batch, n), dtype=torch.long)
        allow = torch.eye(n, dtype=torch.bool).unsqueeze(0).repeat(batch, 1, 1)  # pads attend to themselves only
        spans: list[list[tuple[int, int]]] = []
        for b, st in enumerate(s_ids):
            S = len(st)
            ids[b, :S] = torch.tensor(st)
            pos[b, :S] = torch.arange(S)
            block = torch.ones(S, S, dtype=torch.bool)
            allow[b, :S, :S] = torch.tril(block) if causal else block
            start, row = S, []
            for qn, qi in enumerate(q_ids):
                L = len(qi)
                ids[b, start : start + L] = torch.tensor(qi)
                pos[b, start : start + L] = torch.arange(S, S + L)  # every question starts where the state ends
                allow[b, start : start + L, :S] = True
                qb = torch.ones(L, L, dtype=torch.bool)
                allow[b, start : start + L, start : start + L] = torch.tril(qb) if causal else qb
                if self.readout_mode == "pointer" and self.option_isolation:
                    self._isolate_options(allow[b], pos[b], start, L, S, qn, schema)
                row.append((start, start + L))
                start += L
            spans.append(row)
        ids, pos, allow = ids.to(self.device), pos.to(self.device), allow.to(self.device)

        # Compose the block mask with the backbone's native mask, never substitute it: a custom mask replaces
        # whatever structure the model builds for itself (sliding windows, local/global alternation), and the
        # result runs without error while no longer being the pretrained model.
        config = self.encoder.config
        window = getattr(config, "local_attention", None)
        layer_types = set(getattr(config, "layer_types", None) or [])
        structured = "sliding_attention" in layer_types or bool(getattr(config, "use_sliding_window", False))
        if structured and getattr(config, "model_type", "") != "modernbert":
            raise NotImplementedError(f"{config.model_type} uses sliding-window attention, which the block mask "
                                      "does not compose with yet")
        if getattr(config, "model_type", "") == "modernbert" and window:
            # ModernBERT's local layers were pretrained with a sliding window; rebuild it on position ids
            near = (pos.unsqueeze(2) - pos.unsqueeze(1)).abs() <= window // 2
            mask = {"full_attention": self._additive(allow), "sliding_attention": self._additive(allow & near)}
        else:
            mask = self._additive(allow)
        hidden = self._backbone_hidden(input_ids=ids, attention_mask=mask, position_ids=pos)

        slots = [torch.stack([hidden[b, e - 1] if causal else hidden[b, s:e].mean(0) for s, e in spans[b]])
                 for b in range(batch)]
        if self.readout_mode == "pointer":  # each option's in-context vector: (B, k_f, d) per field
            ends = self._cache[("ends", "questions", schema, self.readout_mode, self.pointer_yesno)]
            self._pointer_opts = [None if ends[i] is None else
                                  torch.stack([hidden[b, [spans[b][i][0] + o for o in ends[i]]] for b in range(batch)])
                                  for i in range(len(ends))]
        return torch.stack(slots)

    def _isolate_options(self, allow_b, pos_b, start, L, S, qn, schema) -> None:
        """Option isolation inside one question block (in place on one row's mask and positions): option lines see
        the state, the question head and themselves only; all start at the same position; the tail (</options>,
        <answer>) sees the head, every option and itself, positioned after the longest option."""
        key = ("questions", schema, self.readout_mode, self.pointer_yesno)
        starts, ends = self._cache[("starts",) + key][qn], self._cache[("ends",) + key][qn]
        if starts is None:  # a yes/no question in the slot layout: nothing to isolate
            return
        h = starts[0]
        spans = [(a, e + 1) for a, e in zip(starts, ends)]
        longest = max(e - a for a, e in spans)
        t0 = spans[-1][1]
        blk = allow_b[start : start + L, start : start + L]
        for a, e in spans:  # an option line: head + itself (causal), no other option
            blk[a:e, h:t0] = False
            blk[a:e, a:e] = torch.tril(torch.ones(e - a, e - a, dtype=torch.bool))
            pos_b[start + a : start + e] = torch.arange(S + h, S + h + (e - a))
        pos_b[start + t0 : start + L] = torch.arange(S + h + longest, S + h + longest + (L - t0))

    def _replicated_slots(self, s_ids: list[list[int]], q_ids: list[list[int]]) -> torch.Tensor:
        """The same view without a custom mask: one right-padded row [state | q_i] per (state, question), under the
        backbone's own causal attention, natural positions (questions start where the state ends). (B, F, d)."""
        if self.pool != "last":
            raise NotImplementedError("one row per question needs a causal backbone")
        pad = getattr(self.tokenizer, "pad_token_id", None) or 0
        rows = [st + qi for st in s_ids for qi in q_ids]
        n = max(len(r) for r in rows)
        ids = torch.full((len(rows), n), pad, dtype=torch.long)
        attn = torch.zeros((len(rows), n), dtype=torch.long)
        for j, r in enumerate(rows):
            ids[j, : len(r)] = torch.tensor(r)
            attn[j, : len(r)] = 1
        ids, attn = ids.to(self.device), attn.to(self.device)
        pos = torch.arange(n, device=self.device).unsqueeze(0).expand(len(rows), n)
        hidden = self._backbone_hidden(input_ids=ids, attention_mask=attn, position_ids=pos)
        out = torch.stack([hidden[j, len(r) - 1] for j, r in enumerate(rows)])
        return out.view(len(s_ids), len(q_ids), -1)

    def _forked_pointer_slots(self, s_ids: list[list[int]], q_ids: list[list[int]], schema: Schema) -> torch.Tensor:
        """Pointer readout on a hybrid backbone by cache forks (see the module docstring). Sets self._pointer_opts
        (per field: (B, k, d) option vectors, or None for a slot-scored yes/no field) and returns the slots (B, F, d).

        The states are left-padded into one batch and run once with the cache on. Every branch row continues from its
        own state's cache: rows are processed in chunks of at most COXLM_FORK_TOKENS tokens (default 16384), each
        chunk forked afresh from the states' cache, so memory stays bounded however many options a question has."""
        if self.pool != "last":
            raise NotImplementedError("cache forks need a causal backbone")
        key = ("questions", schema, self.readout_mode, self.pointer_yesno)
        starts_all, ends_all = self._cache[("starts",) + key], self._cache[("ends",) + key]
        pad = getattr(self.tokenizer, "pad_token_id", None) or 0
        dev = self.device
        nb, S = len(s_ids), max(len(st) for st in s_ids)
        sid = torch.full((nb, S), pad, dtype=torch.long)
        smask = torch.zeros((nb, S), dtype=torch.long)
        spos = torch.zeros((nb, S), dtype=torch.long)
        for b, st in enumerate(s_ids):
            lp = S - len(st)
            sid[b, lp:] = torch.tensor(st)
            smask[b, lp:] = 1
            spos[b, lp:] = torch.arange(len(st))
        sid, smask, spos = sid.to(dev), smask.to(dev), spos.to(dev)
        base = self.encoder(input_ids=sid, attention_mask=smask, position_ids=spos, use_cache=True).past_key_values

        rows, owner, plan = [], [], []
        for q, starts, ends in zip(q_ids, starts_all, ends_all):
            if starts is None:  # a yes/no field in the slot layout: its whole block after the state
                plan.append(("plain", len(rows), None))
                for b in range(nb):
                    rows.append(q)
                    owner.append(b)
                continue
            head, tail = q[: starts[0]], q[ends[-1] + 1:]
            lines = [q[a: e + 1] for a, e in zip(starts, ends)]
            plan.append(("pointer", len(rows), len(lines)))
            for b in range(nb):
                for r in [head + tail] + [head + line for line in lines]:
                    rows.append(r)
                    owner.append(b)

        budget = int(os.environ.get("COXLM_FORK_TOKENS", "16384"))
        per = max(1, budget // max(len(r) for r in rows))
        outs = []
        for i in range(0, len(rows), per):
            chunk = rows[i: i + per]
            own = torch.tensor(owner[i: i + per], dtype=torch.long, device=dev)
            cache = copy.copy(base) if i + per < len(rows) else base  # the last chunk may consume the base cache
            if cache is not base:
                cache.layers = [copy.copy(layer) for layer in base.layers]
                for layer in cache.layers:  # fresh containers: forking replaces tensors, never edits the base's
                    for a in ("conv_states", "recurrent_states", "is_conv_states_initialized",
                              "is_recurrent_states_initialized", "has_previous_state"):
                        v = getattr(layer, a, None)
                        if isinstance(v, (dict, list)):
                            setattr(layer, a, type(v)(v))
            cache.reorder_cache(own)
            n, L = len(chunk), max(len(r) for r in chunk)
            ids = torch.full((n, L), pad, dtype=torch.long)
            m = torch.zeros((n, L), dtype=torch.long)
            pos = torch.zeros((n, L), dtype=torch.long)
            for j, (r, b) in enumerate(zip(chunk, owner[i: i + per])):
                ids[j, : len(r)] = torch.tensor(r)
                m[j, : len(r)] = 1
                pos[j] = len(s_ids[b]) + torch.arange(L)  # branches start where their state ends
            ids, m, pos = ids.to(dev), m.to(dev), pos.to(dev)
            out = self.encoder(input_ids=ids, attention_mask=torch.cat([smask[own], m], dim=1), position_ids=pos,
                               past_key_values=cache, use_cache=True)
            h = self._head_view(out)
            outs.append(h[torch.arange(n, device=dev), torch.tensor([len(r) for r in chunk], device=dev) - 1])
        last = torch.cat(outs)

        slots, opts = [], []
        for kind, start, k in plan:
            if kind == "plain":
                slots.append(last[start: start + nb])
                opts.append(None)
            else:
                blk = last[start: start + nb * (k + 1)].view(nb, k + 1, -1)
                slots.append(blk[:, 0])
                opts.append(blk[:, 1:])
        self._pointer_opts = opts
        return torch.stack(slots, dim=1)

    # -- forward -------------------------------------------------------------

    def forward(self, states: list[str], schema: Schema) -> list[torch.Tensor]:
        """States -> a list (length F) of per-field logit tensors, each (B, k_f)."""
        if self.readout_mode == "pointer" and self.pointer_yesno:
            option_embs = [None] * len(schema)  # every field is scored from its in-context option states
        else:
            option_embs = self._option_embeddings(schema)
        slots = self._packed_slots(states, schema)  # (B, F, d)
        slots_p = self.proj(slots)  # (B, F, proj)
        temp = self.temperature()
        logits: list[torch.Tensor] = []
        opts, self._pointer_opts = getattr(self, "_pointer_opts", None), None
        for i, f in enumerate(schema):
            if self.readout_mode == "pointer" and not (f.kind == "yesno" and not self.pointer_yesno):
                if f.n_outputs != len(f.options):
                    raise NotImplementedError("pointer readout: options outside the listed ones (e.g. none_of_these)")
                q = self.pointer_q(slots[:, i, :].float())  # (B, dp)
                k = self.pointer_k(opts[i].float())  # (B, k_f, dp)
                logits.append(torch.einsum("bd,bkd->bk", q, k) * self._pointer_scale)
                continue
            opt_p, slot = self.proj(option_embs[i]), slots_p[:, i, :]  # (k_f, proj), (B, proj)
            if self.normalize:
                slot, opt_p = F.normalize(slot, dim=-1), F.normalize(opt_p, dim=-1)
            logits.append((slot @ opt_p.t()) / temp)
        return logits

    @torch.no_grad()
    def predict(self, states: list[str], schema: Schema) -> list[dict[str, list[float]]]:
        """States -> list of {field_name: [probabilities]} per example."""
        was_training = self.training
        self.eval()
        probs = [F.softmax(l, dim=-1) for l in self.forward(states, schema)]
        if was_training:
            self.train()
        return [{schema.fields[i].name: probs[i][b].cpu().tolist() for i in range(len(schema))} for b in range(len(states))]

    @overload
    def decide(self, state: Any, questions: type[Q]) -> Q: ...
    @overload
    def decide(self, state: Any, questions: Schema) -> Answers: ...

    def decide(self, state, questions):
        """One state (a string, a dict rendered as "key: value" lines, or a list rendered as numbered lines) ->
        its answers. ``questions`` is a ``questions(...)`` schema or a ``Questions`` subclass; with a subclass the
        answers come back as an instance of it. Every question is answered in the same forward pass,
        independently, with a full distribution, a confidence, and the kind-specific reading."""
        return self._decide([state], questions, stacklevel=3)[0]

    @overload
    def decide_batch(self, states: Iterable[Any], questions: type[Q]) -> list[Q]: ...
    @overload
    def decide_batch(self, states: Iterable[Any], questions: Schema) -> list[Answers]: ...

    def decide_batch(self, states, questions):
        """Several states in one call -> one answers object per state, in order."""
        if isinstance(states, (str, bytes, dict)):
            raise TypeError("decide_batch() takes a list of states; for one state use decide(state, questions)")
        return self._decide(list(states), questions, stacklevel=3)

    @overload
    def decide_iter(self, states: Iterable[Any], questions: type[Q], batch_size: int = 32) -> Iterator[Q]: ...
    @overload
    def decide_iter(self, states: Iterable[Any], questions: Schema, batch_size: int = 32) -> Iterator[Answers]: ...

    def decide_iter(self, states, questions, batch_size=32):
        """Any iterable of states (a generator, a file, a cursor) -> answers one state at a time, in order, sent in
        batches of ``batch_size`` behind the scenes. Nothing is read ahead beyond the current batch."""
        if isinstance(states, (str, bytes, dict)):
            raise TypeError("decide_iter() takes an iterable of states; for one state use decide(state, questions)")
        for batch in iter_batches(states, batch_size):
            yield from self._decide(batch, questions, stacklevel=3)

    def _decide(self, states: list, questions, stacklevel: int) -> list:
        import warnings

        if getattr(self, "closed", False):
            raise RuntimeError("this model was closed; load it again to use it")

        schema = as_schema(questions)
        features = getattr(self, "trained_features", set())
        reads_stems = "channel_ablation" in features
        bare = [f.name for f in schema if f.kind == "yesno" and not (f.descriptions and any(f.descriptions)) and not (reads_stems and f.instructions)]
        if bare:
            warnings.warn(
                f"yes/no field(s) {bare} carry nothing this checkpoint reads. Every yes/no question shares the option "
                "names 'no' and 'yes'; this checkpoint tells them apart only by their option descriptions and ignores "
                "the instructions, so the answers are unreliable and may invert. "
                "Pass criteria={'no': ..., 'yes': ...}, or use a checkpoint that reads instructions.",
                stacklevel=stacklevel,
            )
        unseen_none = [f.name for f in schema if f.none_option] if "none_option" not in features else []
        if unseen_none:
            warnings.warn(
                f"field(s) {unseen_none} offer none_of_these, but this checkpoint was not trained with the none option. "
                "To it that is an untrained option: it will rarely be chosen even when nothing offered fits. "
                "Remove none_option, or use a checkpoint trained with it.",
                stacklevel=stacklevel,
            )
        preds = self.predict([render_state(s) for s in states], schema)
        return [answers_for(questions, p) for p in preds]


CAUSAL_MODEL_TYPES = {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "gemma3", "gemma3_text", "phi3", "olmo2", "smollm3",
                      "qwen3_5", "qwen3_5_text", "granite", "granitemoehybrid", "olmo3", "ministral3", "mistral3", "nemotron_h"}
# backbones whose layers cannot take a custom attention mask (recurrent / linear attention): one row per question
HYBRID_MODEL_TYPES = {"qwen3_5", "qwen3_5_text", "granitemoehybrid", "nemotron_h"}
MAMBA_MODEL_TYPES = {"granitemoehybrid", "nemotron_h"}
# multimodal checkpoints: the text model is the backbone (the vision tower is dropped)
MULTIMODAL_TEXT = {"qwen3_5": "language_model", "mistral3": "language_model"}


def _enable_hub_conv1d() -> None:
    """Qwen3.5's linear-attention layers fall back to a slow PyTorch conv unless a package named causal_conv1d
    imports. coxlm/_shims/causal_conv1d forwards to the Hugging Face hub build (no compilation); matches the reference within
    bf16 (hidden rel 0.9%), ~1.1x faster on a 2B. Off with COX_HUB_CONV1D=0 or without `kernels`."""
    import importlib.util
    shims = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shims")
    if os.environ.get("COX_HUB_CONV1D", "1") != "0" and importlib.util.find_spec("kernels") and shims not in sys.path:
        sys.path.insert(0, shims)


def _no_init_weights():
    """transformers' context that skips random initialisation (moved between versions); a no-op if neither exists."""
    from contextlib import nullcontext

    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:  # pragma: no cover
            return nullcontext()
    return no_init_weights()


def build_model(
    encoder_name: str,
    dtype: str | None = None,
    pool: str | None = None,
    state_norm: bool | str | None = None,
    lora_r: int = 0,
    lora_alpha: int | None = None,
    pretrained_weights: bool = True,
    **kwargs,
) -> AmortizedDecisionModel:
    """Build the model on a pretrained backbone (downloads on first use), ready for a checkpoint's weights.

    Any model that returns per-token hidden states works: a bidirectional encoder (ModernBERT) or a decoder LLM used
    as an encoder (Qwen3, Qwen3.5). ``dtype`` "bf16" loads the backbone in bfloat16 (the head stays fp32); ``pool``
    defaults to "last" for causal model types and "mean" otherwise. ``lora_r`` > 0 attaches empty LoRA adapters for
    an adapter checkpoint to fill. ``pretrained_weights=False`` builds the backbone from its configuration alone
    (no weight download, no initialisation), for a full-weight checkpoint that supplies every backbone weight.
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
    if pretrained_weights:
        encoder = AutoModel.from_pretrained(encoder_name, **load_kw)
    else:
        with _no_init_weights():
            encoder = AutoModel.from_config(config, dtype=torch_dtype) if torch_dtype else AutoModel.from_config(config)
    if config.model_type in MULTIMODAL_TEXT:
        # load the whole checkpoint (so every weight name matches), then keep only the text model
        encoder = getattr(encoder, MULTIMODAL_TEXT[config.model_type])
    model = AmortizedDecisionModel(encoder, tokenizer, hidden_size=encoder.config.hidden_size, pool=pool,
                                   state_norm=state_norm, **kwargs)
    if lora_r > 0:
        from peft import LoraConfig, get_peft_model

        # Mamba mixers: PEFT refuses LoRA on their out_proj / conv1d
        exclude = ["out_proj", "conv1d"] if config.model_type in MAMBA_MODEL_TYPES else None
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha or 2 * lora_r, lora_dropout=0.0, target_modules="all-linear",
                         exclude_modules=exclude)
        model.encoder = get_peft_model(model.encoder, cfg)
        for n_, p_ in model.encoder.named_parameters():
            if "lora_" in n_:
                p_.data = p_.data.float()  # adapters in fp32 even on a bf16 base, as trained
    return model
