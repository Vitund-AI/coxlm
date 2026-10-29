"""A tiny random cox-1-style model export for CPU tests (no GPU, no download): shared by the MLX parity tests and the
fine-tune tests."""
from __future__ import annotations

import json
import os

WORDS = ("my and I state question options option answer name the a is it this message about card refund billing support sales "
         "team urgency low medium high yes no spam genuine order money charged twice want back today crash app "
         "settings open which handles how urgent customer asking for one two three four five six seven eight nine "
         "ten eleven twelve contract says statement true false").split()


def _tokenizer(folder: str) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<unk>": 1}
    for w in WORDS + list("<>/=\"\n.,!?:;-_'0123456789"):
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>").save_pretrained(folder)


def make_tiny_export(folder: str, seed: int = 0) -> str:
    """A cox-1-style export folder: a 4-block Qwen3.5 text model (3 Gated DeltaNet + 1 gated attention), random
    weights (norms and decay parameters included), and a random pointer head. ``seed`` changes the weights."""
    import torch
    import transformers
    from safetensors.torch import save_file

    _tokenizer(folder)
    cfg = transformers.Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, tie_word_embeddings=True,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    torch.manual_seed(seed)
    model = transformers.Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.normal_(0.0, 0.2)  # zero-centred norms: exercise the (1 + w)
            elif name.endswith(("A_log", "dt_bias")):
                p.uniform_(-1.0, 1.0)
            else:
                p.normal_(0.0, 0.08)
    model.save_pretrained(folder)  # model.safetensors with "model."-prefixed keys, as the real exports
    d, dp = 64, 16
    g = torch.Generator().manual_seed(seed + 1)
    head = {"log_temp": torch.tensor(-2.0), "pointer_q.weight": torch.randn(dp, d, generator=g) * 0.3,
            "pointer_q.bias": torch.randn(dp, generator=g) * 0.1, "pointer_k.weight": torch.randn(dp, d, generator=g) * 0.3,
            "pointer_k.bias": torch.randn(dp, generator=g) * 0.1, "proj.weight": torch.randn(d, d, generator=g) * 0.2,
            "state_mu": torch.randn(d, generator=g) * 0.1, "state_sd": torch.rand(d, generator=g) + 0.5,
            "state_calibrated": torch.tensor(True)}
    save_file(head, os.path.join(folder, "cox_head.safetensors"))
    build = {"state_norm": "standardize", "question_mode": "packed", "prompt_format": "xml", "readout_mode": "pointer",
             "option_isolation": True, "pointer_yesno": False, "yesno_layout": "slot", "pointer_dim": dp}
    json.dump({"name": "tiny", "format_version": 1, "final_norm": "identity", "base_model": "tiny",
               "capabilities": {"reads_instructions": True}, "meta": {"build": build, "features": []}},
              open(os.path.join(folder, "cox_config.json"), "w"))
    return folder


def quantize_rows(w):
    """int8wo quantization of a 2-D weight, as the export builder does it (torchao Int8Tensor PerRow): the weight in
    bf16, scale = absmax / 127.5 computed in bf16, q8 = round(W / scale) clamped to int8."""
    import torch

    w = w.to(torch.bfloat16)
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.5).clamp(min=torch.finfo(torch.float32).eps).to(torch.bfloat16)
    q8 = torch.clamp(torch.round(w.float() / scale.float()), -128, 127).to(torch.int8)
    return q8.contiguous(), scale.contiguous()


def make_int8_exports(src: str, int8_dir: str, ref_dir: str) -> tuple[str, str]:
    """From an export: an int8wo export (every decoder-block Linear weight and the embedding int8) and a reference
    export holding exactly the dequantized values (bf16) in the same places, every other tensor unchanged."""
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    w = load_file(os.path.join(src, "model.safetensors"))
    q, ref = {}, {}
    for k, v in w.items():
        if v.ndim == 2 and (".layers." in k or k.endswith("embed_tokens.weight")):
            q8, scale = quantize_rows(v)
            q[k + ".q8"], q[k + ".scale"] = q8, scale
            ref[k] = (q8.float() * scale.float()).to(torch.bfloat16)
        else:
            q[k] = ref[k] = v
    for d, tensors in ((int8_dir, q), (ref_dir, ref)):
        os.makedirs(d, exist_ok=True)
        for name in os.listdir(src):
            if name != "model.safetensors":
                shutil.copy2(os.path.join(src, name), os.path.join(d, name))
        save_file(tensors, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(src, "cox_config.json")))
    cfg["format_version"] = 2
    cfg["weights"] = {"format": "int8wo", "version": 1, "embed": "int8", "min_coxlm": "0.7.0"}
    json.dump(cfg, open(os.path.join(int8_dir, "cox_config.json"), "w"))
    return int8_dir, ref_dir
