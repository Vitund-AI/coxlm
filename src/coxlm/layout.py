"""Prompt layout and the decide API, shared by every backend (PyTorch in model.py, MLX in mlx_backend.py).

No torch or MLX here: a backend supplies ``tokenizer``, ``max_length``, ``pool``, ``readout_mode``, ``pointer_yesno``,
``trained_features``, a ``_cache`` dict and ``predict(states, schema)``; this mixin writes the XML prompt, tokenises it
(memoised) and turns predictions into typed answers.
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, overload

from .decide import Answers, Q, answers_for, as_schema, iter_batches, render_state
from .schema import Schema


class PromptLayout:
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


    # -- the decide API --------------------------------------------------------

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
