"""
Minimal attention debugging utilities for PETR decoders.

Usage (example):
    from mmcv.models.utils.petr_transformers import PETRTransformerDecoder
    from tools.attn_debug import debug_forward_decoder

    # Build decoder as usual (ideally with flash_attn=False for cross-attn)
    decoder = PETRTransformerDecoder(..., flash_attn=False)

    out = debug_forward_decoder(
        decoder,
        query, key,
        query_pos=query_pos, key_pos=key_pos,
        attn_mask=attn_mask,
        temp_memory=temp_memory, temp_pos=temp_pos,
        return_attn=True,
    )

    queries = out["queries"]          # [L, B, Nq, C]
    cross_attn = out["cross_attn"]     # [L, B, Nq, Nk] or None
    self_attn = out["self_attn"]       # [L, B, Nq, Nkv] or None

Note:
  - If your decoder is built with flash_attn=True for cross-attention, the
    returned attention weights may be None. For debugging purposes, construct
    the decoder with flash_attn=False.
  - These helpers do NOT change your model; they execute a debug forward that
    mirrors decoder-layer logic and return attention maps when available.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


def _to_torch_mha_from_flash(
    flash_mha: nn.Module,
    embed_dim: int,
    num_heads: int,
    dropout: float,
) -> nn.MultiheadAttention:
    """Create a torch.nn.MultiheadAttention and copy weights from a FlashMHA-like module.

    This expects the source module to have attributes:
      - in_proj_weight (Parameter [3*E, E])
      - in_proj_bias   (Parameter [3*E]) or None
      - out_proj       (Linear[E,E]) with .weight and .bias
    """
    mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    # Copy projections if present
    if hasattr(flash_mha, "in_proj_weight") and flash_mha.in_proj_weight is not None:
        with torch.no_grad():
            mha.in_proj_weight.copy_(flash_mha.in_proj_weight)
    if hasattr(flash_mha, "in_proj_bias") and flash_mha.in_proj_bias is not None and mha.in_proj_bias is not None:
        with torch.no_grad():
            mha.in_proj_bias.copy_(flash_mha.in_proj_bias)
    if hasattr(flash_mha, "out_proj") and isinstance(flash_mha.out_proj, nn.Linear):
        with torch.no_grad():
            mha.out_proj.weight.copy_(flash_mha.out_proj.weight)
            if mha.out_proj.bias is not None and flash_mha.out_proj.bias is not None:
                mha.out_proj.bias.copy_(flash_mha.out_proj.bias)
    return mha


def _prepare_attn_module(attn_wrapper: nn.Module, embed_dim: int, num_heads: int, dropout: float) -> Tuple[nn.MultiheadAttention, nn.Module]:
    """Resolve an attention module from a MultiHeadAttentionwDropout-like wrapper.

    Returns (torch_mha, proj_drop) where:
      - torch_mha: a nn.MultiheadAttention module that returns attention weights
      - proj_drop: the dropout module applied to the attention output

    If the underlying module is already nn.MultiheadAttention, it is used directly.
    If it is a FlashMHA-like module, we create a torch MHA and copy the weights.
    """
    # attn_wrapper is expected to have attributes:
    #   - attn: either nn.MultiheadAttention or a FlashMHA-like module
    #   - proj_drop: nn.Dropout
    attn = getattr(attn_wrapper, "attn", None)
    proj_drop = getattr(attn_wrapper, "proj_drop", nn.Dropout(dropout))

    if isinstance(attn, nn.MultiheadAttention):
        return attn, proj_drop
    # Fallback: convert from flash to torch MHA
    torch_mha = _to_torch_mha_from_flash(attn, embed_dim, num_heads, dropout)
    return torch_mha, proj_drop


def _run_attn(
    mha: nn.MultiheadAttention,
    proj_drop: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_pos: Optional[torch.Tensor],
    key_pos: Optional[torch.Tensor],
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a single MHA + dropout + residual with positional encodings.

    All tensors are expected in batch_first format: [B, L, C].
    Returns (out_with_residual, attn_weights[B, Lq, Lk]).
    """
    q = query + query_pos if query_pos is not None else query
    k = key + key_pos if key_pos is not None else key
    out, attn = mha(query=q, key=k, value=value, attn_mask=attn_mask)
    out = proj_drop(out)
    return out + query, attn


def debug_forward_decoder_layer(
    decoder_layer: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    query_pos: Optional[torch.Tensor],
    key_pos: Optional[torch.Tensor],
    attn_mask: Optional[torch.Tensor],
    temp_memory: Optional[torch.Tensor] = None,
    temp_pos: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    """Mirror PETRTransformerDecoderLayer forward but always return attention weights.

    decoder_layer.transformer_layers is expected as:
      [ self_attn_wrap, LN1, cross_attn_wrap, LN2, FFN, LN3 ]
    """
    layers = getattr(decoder_layer, "transformer_layers")
    embed_dims = getattr(decoder_layer, "_embed_dims", query.shape[-1])
    # Identify submodules
    self_attn_wrap = layers[0]
    ln1 = layers[1]
    cross_attn_wrap = layers[2]
    ln2 = layers[3]
    ffn = layers[4]
    ln3 = layers[5]

    num_heads = getattr(decoder_layer, "_num_heads", 8)
    dropout = getattr(self_attn_wrap, "_dropout", 0.0)

    # Prepare self-attention K/V (concat temporal memory if provided)
    if temp_memory is not None:
        temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
        temp_pos_all = torch.cat([query_pos, temp_pos], dim=1) if (query_pos is not None and temp_pos is not None) else None
    else:
        temp_key = temp_value = query
        temp_pos_all = query_pos

    self_mha, self_proj_drop = _prepare_attn_module(self_attn_wrap, embed_dims, num_heads, dropout)
    query, attn0 = _run_attn(self_mha, self_proj_drop, query, temp_key, temp_value, query_pos, temp_pos_all, attn_mask)
    query = ln1(query)

    # Cross attention
    cross_dropout = getattr(cross_attn_wrap, "_dropout", 0.0)
    cross_mha, cross_proj_drop = _prepare_attn_module(cross_attn_wrap, embed_dims, num_heads, cross_dropout)
    query, attn1 = _run_attn(cross_mha, cross_proj_drop, query, key, key, query_pos, key_pos, None)
    query = ln2(query)

    # FFN + LN
    query = ffn(query)
    query = ln3(query)

    return query, {"self_attn": attn0, "cross_attn": attn1}


def debug_forward_decoder(
    decoder: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    query_pos: Optional[torch.Tensor] = None,
    key_pos: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    temp_memory: Optional[torch.Tensor] = None,
    temp_pos: Optional[torch.Tensor] = None,
    return_attn: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    """Run an entire PETRTransformerDecoder and collect attention maps per layer.

    Returns a dict with keys:
      - queries:   [L, B, Nq, C]
      - self_attn: [L, B, Nq, Nkv] or None
      - cross_attn:[L, B, Nq, Nk] or None
    """
    out_queries: List[torch.Tensor] = []
    self_list: List[Optional[torch.Tensor]] = []
    cross_list: List[Optional[torch.Tensor]] = []

    layers = getattr(decoder, "_layers")
    for layer in layers:
        query, attn = debug_forward_decoder_layer(
            layer, query, key, query_pos, key_pos, attn_mask, temp_memory, temp_pos
        )
        out_queries.append(query)
        self_list.append(attn.get("self_attn", None))
        cross_list.append(attn.get("cross_attn", None))

    stacked_queries = torch.stack(out_queries, dim=0)

    def _stack_or_none(items):
        if all(x is None for x in items):
            return None
        valid = [x for x in items if x is not None]
        try:
            return torch.stack(valid, dim=0)
        except Exception:
            return None

    ret = {"queries": stacked_queries}
    if return_attn:
        ret["self_attn"] = _stack_or_none(self_list)
        ret["cross_attn"] = _stack_or_none(cross_list)
    return ret

