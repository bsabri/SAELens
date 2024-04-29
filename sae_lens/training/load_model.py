from typing import Any, cast

import einops
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookedRootModule
from transformers import AutoTokenizer

n_heads = 16
n_layers = 1
d_model = 1024

cfg = HookedTransformerConfig(
    n_layers=n_layers,
    d_model=d_model,
    eps=1e-6,
    d_head=int(d_model / n_heads),
    # d_head=16,
    n_heads=n_heads,
    n_key_value_heads = 16,
    d_mlp=4096,
    d_vocab=32000,
    n_ctx=1024,
    act_fn="silu",
    normalization_type="RMS",
    positional_embedding_type="rotary",
    rotary_adjacent_pairs=False,
    final_rms=True,
    gated_mlp=True,
    rotary_dim= d_model // n_heads,
    )

def convert_llama_weights(llama: dict, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = llama["model.embed_tokens.weight"]

    # Some models with the Llama architecture use Grouped Query Attention, and so for these we need to modify
    # the state dict keys for the K/V attention weight/biases, prepending "_" to the key names.
    using_gqa = cfg.n_key_value_heads is not None
    gqa_uscore = "_" if using_gqa else ""

    # llama has no biases anywhere and deals with everything else roughly like
    # GPTNeoX with different names

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = llama[f"model.layers.{l}.input_layernorm.weight"]

        W_Q = llama[f"model.layers.{l}.self_attn.q_proj.weight"]
        W_K = llama[f"model.layers.{l}.self_attn.k_proj.weight"]
        W_V = llama[f"model.layers.{l}.self_attn.v_proj.weight"]
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(
            W_K, "(n h) m->n m h", n=cfg.n_key_value_heads if using_gqa else cfg.n_heads
        )
        W_V = einops.rearrange(
            W_V, "(n h) m->n m h", n=cfg.n_key_value_heads if using_gqa else cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            cfg.n_key_value_heads if using_gqa else cfg.n_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            cfg.n_key_value_heads if using_gqa else cfg.n_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )

        W_O = llama[f"model.layers.{l}.self_attn.o_proj.weight"]
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O.to(device=cfg.device)

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

        state_dict[f"blocks.{l}.ln2.w"] = llama[f"model.layers.{l}.post_attention_layernorm.weight"]

        state_dict[f"blocks.{l}.mlp.W_in"] = llama[f"model.layers.{l}.mlp.up_proj.weight"].T
        state_dict[f"blocks.{l}.mlp.W_gate"] = llama[f"model.layers.{l}.mlp.gate_proj.weight"].T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=cfg.device
        )

        state_dict[f"blocks.{l}.mlp.W_out"] = llama[f"model.layers.{l}.mlp.down_proj.weight"].T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )
    state_dict["ln_final.w"] = llama["model.norm.weight"]

    state_dict["unembed.W_U"] = llama["lm_head.weight"].T
    state_dict["unembed.b_U"] = torch.zeros(
        cfg.d_vocab, dtype=cfg.dtype, device=cfg.device
    )

    return state_dict



def load_model(
    model_class_name: str,
    model_name: str,
    device: str | torch.device | None = None,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
) -> HookedRootModule:
    
    if model_class_name == "HookedTransformer":
        home_path = "/export/home/sboughorbel/codes/tinystories/"
        checkpoint = torch.load(home_path + model_name +"/pytorch_model.bin", map_location=device)
        model = HookedTransformer(cfg)
        model.to(device)
        model.load_and_process_state_dict(convert_llama_weights(checkpoint, cfg), fold_ln=False, center_unembed=False, center_writing_weights=False)
        # return HookedTransformer.from_pretrained(model_name=model_name, device=device)
        tokenizer = AutoTokenizer.from_pretrained(home_path + model_name) 
        model.set_tokenizer(tokenizer)
        return model 

    elif model_class_name == "HookedMamba":
        try:
            from mamba_lens import HookedMamba
        except ImportError:
            raise ValueError(
                "mamba-lens must be installed to work with mamba models. This can be added with `pip install sae-lens[mamba]`"
            )
        # HookedMamba has incorrect typing information, so we need to cast the type here
        return cast(
            HookedRootModule,
            HookedMamba.from_pretrained(
                model_name, device=cast(Any, device), **model_from_pretrained_kwargs
            ),
        )
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")
