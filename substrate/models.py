"""Segmented torch2jax execution engine for frozen LLM transformer blocks.

Converts Hugging Face / PyTorch transformer segments into JAX-native computations
using samuela/torch2jax while maintaining explicit JAX interception and modification.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from torch2jax import t2j_module

from .architecture import Architecture


# ── Strict State-Dict Validation ─────────────────────────────────────────────

def validate_segment_state_dict(
    segment_name: str,
    module: nn.Module,
    supplied_sd: Mapping[str, Any],
) -> None:
    """Assert exact key equality between PyTorch module state_dict and JAX dictionary."""
    expected_keys = set(module.state_dict().keys())
    actual_keys = set(supplied_sd.keys())

    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys

    if missing or unexpected:
        error_msg = [f"State-dict mismatch for segment '{segment_name}':"]
        if missing:
            error_msg.append(f"  Missing keys ({len(missing)}): {sorted(missing)}")
        if unexpected:
            error_msg.append(f"  Unexpected keys ({len(unexpected)}): {sorted(unexpected)}")
        raise KeyError("\n".join(error_msg))


# ── GPT-2 Contiguous Segment Wrappers ────────────────────────────────────────

class GPT2InitialSegment(nn.Module):
    """GPT-2 Segment 0: Embeddings + Blocks [0 .. end_layer]."""

    def __init__(self, wte: nn.Embedding, wpe: nn.Embedding, blocks: Sequence[nn.Module]):
        super().__init__()
        self.wte = wte
        self.wpe = wpe
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        h = self.wte(input_ids) + self.wpe(position_ids)
        for block in self.blocks:
            h = block(h)[0]
        return h


class GPT2MiddleSegment(nn.Module):
    """GPT-2 Segment k: Blocks [start_layer .. end_layer]."""

    def __init__(self, blocks: Sequence[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        h = hidden_state
        for block in self.blocks:
            h = block(h)[0]
        return h


class GPT2FinalSegment(nn.Module):
    """GPT-2 Final Segment: Blocks [start_layer .. L-1] + ln_f + lm_head."""

    def __init__(self, blocks: Sequence[nn.Module], ln_f: nn.LayerNorm, lm_head: nn.Linear):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = ln_f
        self.lm_head = lm_head

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        h = hidden_state
        for block in self.blocks:
            h = block(h)[0]
        h = self.ln_f(h)
        return self.lm_head(h)


class GPT2MonolithicSegment(nn.Module):
    """GPT-2 Monolithic Segment: Embeddings + All Blocks + ln_f + lm_head."""

    def __init__(
        self,
        wte: nn.Embedding,
        wpe: nn.Embedding,
        blocks: Sequence[nn.Module],
        ln_f: nn.LayerNorm,
        lm_head: nn.Linear,
    ):
        super().__init__()
        self.wte = wte
        self.wpe = wpe
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = ln_f
        self.lm_head = lm_head

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        h = self.wte(input_ids) + self.wpe(position_ids)
        for block in self.blocks:
            h = block(h)[0]
        h = self.ln_f(h)
        return self.lm_head(h)


# ── GPT-NeoX / Pythia Contiguous Segment Wrappers ────────────────────────────

class NeoXInitialSegment(nn.Module):
    """NeoX Segment 0: embed_in + global rotary_emb + Blocks [0 .. end_layer]."""

    def __init__(self, embed_in: nn.Embedding, rotary_emb: nn.Module, layers: Sequence[nn.Module]):
        super().__init__()
        self.embed_in = embed_in
        self.rotary_emb = rotary_emb
        self.layers = nn.ModuleList(layers)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h = self.embed_in(input_ids)
        cos, sin = self.rotary_emb(h, position_ids)
        rotary_pos_emb = (cos, sin)

        B, T = input_ids.shape
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1)
        attn_mask = causal_mask.view(1, 1, T, T).expand(B, 1, T, T)

        for layer in self.layers:
            h = layer(h, attention_mask=attn_mask, position_embeddings=rotary_pos_emb)[0]

        return h, rotary_pos_emb


class NeoXMiddleSegment(nn.Module):
    """NeoX Segment k: Blocks [start_layer .. end_layer]."""

    def __init__(self, layers: Sequence[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_state: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_state.shape
        rotary_pos_emb = (cos, sin)
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=hidden_state.device), diagonal=1)
        attn_mask = causal_mask.view(1, 1, T, T).expand(B, 1, T, T)

        h = hidden_state
        for layer in self.layers:
            h = layer(h, attention_mask=attn_mask, position_embeddings=rotary_pos_emb)[0]
        return h


class NeoXFinalSegment(nn.Module):
    """NeoX Final Segment: Blocks [start_layer .. L-1] + final_layer_norm + embed_out."""

    def __init__(self, layers: Sequence[nn.Module], final_layer_norm: nn.LayerNorm, embed_out: nn.Linear):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.final_layer_norm = final_layer_norm
        self.embed_out = embed_out

    def forward(self, hidden_state: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_state.shape
        rotary_pos_emb = (cos, sin)
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=hidden_state.device), diagonal=1)
        attn_mask = causal_mask.view(1, 1, T, T).expand(B, 1, T, T)

        h = hidden_state
        for layer in self.layers:
            h = layer(h, attention_mask=attn_mask, position_embeddings=rotary_pos_emb)[0]
        h = self.final_layer_norm(h)
        return self.embed_out(h)


class NeoXMonolithicSegment(nn.Module):
    """NeoX Monolithic Segment: embed_in + rotary_emb + All Layers + final_layer_norm + embed_out."""

    def __init__(
        self,
        embed_in: nn.Embedding,
        rotary_emb: nn.Module,
        layers: Sequence[nn.Module],
        final_layer_norm: nn.LayerNorm,
        embed_out: nn.Linear,
    ):
        super().__init__()
        self.embed_in = embed_in
        self.rotary_emb = rotary_emb
        self.layers = nn.ModuleList(layers)
        self.final_layer_norm = final_layer_norm
        self.embed_out = embed_out

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        h = self.embed_in(input_ids)
        cos, sin = self.rotary_emb(h, position_ids)
        rotary_pos_emb = (cos, sin)

        B, T = input_ids.shape
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1)
        attn_mask = causal_mask.view(1, 1, T, T).expand(B, 1, T, T)

        for layer in self.layers:
            h = layer(h, attention_mask=attn_mask, position_embeddings=rotary_pos_emb)[0]
        h = self.final_layer_norm(h)
        return self.embed_out(h)


# ── Segmented Engine Class ───────────────────────────────────────────────────

class SegmentedTorch2JaxEngine:
    """Partitions PyTorch transformer into K+1 segments and converts them to JAX."""

    def __init__(
        self,
        torch_model: nn.Module,
        arch: Architecture | None = None,
        intercept_layers: Sequence[int] | None = None,
    ):
        torch_model.eval()
        if arch is None:
            #auto-architecture detection 
            from .loader import state_dict_to_jax_pytree
            from .architecture import detect_architecture
            params_pytree = state_dict_to_jax_pytree(torch_model.state_dict())
            arch = detect_architecture(params_pytree, getattr(torch_model, "config", None))

        self.arch = arch
        self.family = arch.model_family
        self.num_layers = arch.num_layers

        # Clean and sort interception points
        self.intercept_layers = tuple(sorted(set(intercept_layers or ())))
        for lyr in self.intercept_layers:
            if lyr < 0 or lyr >= self.num_layers:
                raise ValueError(f"Interception layer {lyr} out of bounds (0..{self.num_layers - 1})")

        # Build PyTorch segments
        self.segments: list[nn.Module] = []
        self.segment_names: list[str] = []
        self._build_segments(torch_model)

        # Convert each segment to JAX via torch2jax
        self.t2j_segments = [t2j_module(seg) for seg in self.segments]

    def _build_segments(self, torch_model: nn.Module) -> None:
        if self.family == "gpt2":
            self._build_gpt2_segments(torch_model)
        elif self.family == "neox":
            self._build_neox_segments(torch_model)
        else:
            raise ValueError(f"Unsupported model family: {self.family}")

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        curr = model

        while True:
            if isinstance(
                curr,
                (nn.DataParallel, nn.parallel.DistributedDataParallel),
            ):
                curr = curr.module
                continue

            # Only use this if your supported model wrappers require it.
            base_model = getattr(curr, "base_model", None)
            if isinstance(base_model, nn.Module) and base_model is not curr:
                curr = base_model
                continue

            break

        return curr

    def _build_gpt2_segments(self, model: nn.Module) -> None:
        model = self._unwrap_model(model)
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            transformer = model

        # Ensure blocks are located
        if not hasattr(transformer, "h") or transformer.h is None:
            if hasattr(model, "h") and model.h is not None:
                transformer = model
            else:
                raise AttributeError(
                    f"Could not locate transformer blocks in {type(model).__name__}. "
                    "Expected 'transformer.h' or 'h' containing a sequence of transformer blocks."
                )

        blocks = list(transformer.h)
        wte = getattr(transformer, "wte", None) or getattr(model, "wte", None)
        wpe = getattr(transformer, "wpe", None) or getattr(model, "wpe", None)
        ln_f = getattr(transformer, "ln_f", None) or getattr(model, "ln_f", None)

        if wte is None or wpe is None or ln_f is None:
            raise AttributeError(
                f"Missing components in GPT-2 model {type(model).__name__}: "
                f"wte={wte is not None}, wpe={wpe is not None}, ln_f={ln_f is not None}"
            )

        lm_head = getattr(model, "lm_head", None) or getattr(transformer, "lm_head", None)
        if lm_head is None:
            lm_head = nn.Linear(wte.weight.shape[1], wte.weight.shape[0], bias=False)
            lm_head.weight = wte.weight

        if len(self.intercept_layers) == 0:
            self.segments.append(GPT2MonolithicSegment(wte, wpe, blocks, ln_f, lm_head))
            self.segment_names = ["monolithic_gpt2"]
            return

        # Segment 0: Embeddings + Blocks [0 .. intercept_layers[0]]
        first_cut = self.intercept_layers[0]
        self.segments.append(GPT2InitialSegment(wte, wpe, blocks[: first_cut + 1]))
        self.segment_names.append(f"seg0_emb_to_block{first_cut}")

        # Middle Segments: Blocks (intercept_layers[i-1] .. intercept_layers[i]]
        for i in range(1, len(self.intercept_layers)):
            p_cut, c_cut = self.intercept_layers[i - 1], self.intercept_layers[i]
            self.segments.append(GPT2MiddleSegment(blocks[p_cut + 1 : c_cut + 1]))
            self.segment_names.append(f"seg{i}_block{p_cut + 1}_to_block{c_cut}")

        # Final Segment: Blocks (intercept_layers[-1] .. L-1] + ln_f + lm_head
        last_cut = self.intercept_layers[-1]
        self.segments.append(GPT2FinalSegment(blocks[last_cut + 1 :], ln_f, lm_head))
        self.segment_names.append(f"seg{len(self.intercept_layers)}_block{last_cut + 1}_to_head")

    def _build_neox_segments(self, model: nn.Module) -> None:
        model = self._unwrap_model(model)
        neox = getattr(model, "gpt_neox", None)
        if neox is None:
            neox = model

        if not hasattr(neox, "layers") or neox.layers is None:
            if hasattr(model, "layers") and model.layers is not None:
                neox = model
            else:
                raise AttributeError(
                    f"Could not locate transformer layers in {type(model).__name__}. "
                    "Expected 'gpt_neox.layers' or 'layers' containing a sequence of transformer layers."
                )

        layers = list(neox.layers)
        embed_in = getattr(neox, "embed_in", None) or getattr(model, "embed_in", None)
        rotary_emb = getattr(neox, "rotary_emb", None) or getattr(model, "rotary_emb", None)
        final_norm = getattr(neox, "final_layer_norm", None) or getattr(model, "final_layer_norm", None)

        if embed_in is None or rotary_emb is None or final_norm is None:
            raise AttributeError(
                f"Missing components in NeoX model {type(model).__name__}: "
                f"embed_in={embed_in is not None}, rotary_emb={rotary_emb is not None}, "
                f"final_layer_norm={final_norm is not None}"
            )

        embed_out = getattr(model, "embed_out", None) or getattr(model, "lm_head", None) or getattr(neox, "embed_out", None)
        if embed_out is None:
            raise KeyError("No output projection found (expected embed_out or lm_head).")

        if len(self.intercept_layers) == 0:
            self.segments.append(
                NeoXMonolithicSegment(embed_in, rotary_emb, layers, final_norm, embed_out)
            )
            self.segment_names = ["monolithic_neox"]
            return

        # Segment 0
        first_cut = self.intercept_layers[0]
        self.segments.append(NeoXInitialSegment(embed_in, rotary_emb, layers[: first_cut + 1]))
        self.segment_names.append(f"neox_seg0_to_block{first_cut}")

        # Middle Segments
        for i in range(1, len(self.intercept_layers)):
            p_cut, c_cut = self.intercept_layers[i - 1], self.intercept_layers[i]
            self.segments.append(NeoXMiddleSegment(layers[p_cut + 1 : c_cut + 1]))
            self.segment_names.append(f"neox_seg{i}_block{p_cut + 1}_to_block{c_cut}")

        # Final Segment
        last_cut = self.intercept_layers[-1]
        self.segments.append(NeoXFinalSegment(layers[last_cut + 1 :], final_norm, embed_out))
        self.segment_names.append(f"neox_seg{len(self.intercept_layers)}_block{last_cut + 1}_to_head")

    def extract_and_validate_params(self) -> dict[str, dict[str, jax.Array]]:
        """Extract state_dicts as float32 JAX arrays and validate exact key parity."""
        params: dict[str, dict[str, jax.Array]] = {}
        for name, seg in zip(self.segment_names, self.segments):
            sd = seg.state_dict()
            jax_sd = {
                k: jnp.asarray(v.detach().cpu().numpy(), dtype=jnp.float32)
                for k, v in sd.items()
            }
            validate_segment_state_dict(name, seg, jax_sd)
            params[name] = jax_sd
        return params

    def run_forward(
        self,
        params: Mapping[str, Any],
        input_ids: jax.Array,
        position_ids: jax.Array | None = None,
        modify_fn: Callable[[jax.Array, int], jax.Array] | None = None,
    ) -> tuple[jax.Array, dict[int, jax.Array]]:
        """Executes the segmented pipeline in JAX, intercepting at layer boundaries."""
        B, T = input_ids.shape
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))

        intermediates: dict[int, jax.Array] = {}

        if self.family == "gpt2":
            seg0_sd = params[self.segment_names[0]]

            # Zero-interception: run monolithic segment directly to logits
            if len(self.intercept_layers) == 0:
                logits = self.t2j_segments[0](input_ids, position_ids, state_dict=seg0_sd)
                return logits, {}

            # 1. Segment 0
            h = self.t2j_segments[0](input_ids, position_ids, state_dict=seg0_sd)
            first_layer = self.intercept_layers[0]
            intermediates[first_layer] = h
            if modify_fn is not None:
                h = modify_fn(h, first_layer)

            # 2. Middle Segments
            for i in range(1, len(self.intercept_layers)):
                seg_sd = params[self.segment_names[i]]
                h = self.t2j_segments[i](h, state_dict=seg_sd)
                layer_idx = self.intercept_layers[i]
                intermediates[layer_idx] = h
                if modify_fn is not None:
                    h = modify_fn(h, layer_idx)

            # 3. Final Segment
            final_sd = params[self.segment_names[-1]]
            logits = self.t2j_segments[-1](h, state_dict=final_sd)
            return logits, intermediates

        else:  # neox
            seg0_sd = params[self.segment_names[0]]

            # Zero-interception: run monolithic segment directly to logits
            if len(self.intercept_layers) == 0:
                logits = self.t2j_segments[0](input_ids, position_ids, state_dict=seg0_sd)
                return logits, {}

            # 1. Segment 0 (returns hidden state + rotary embeddings)
            h, (cos, sin) = self.t2j_segments[0](input_ids, position_ids, state_dict=seg0_sd)
            first_layer = self.intercept_layers[0]
            intermediates[first_layer] = h
            if modify_fn is not None:
                h = modify_fn(h, first_layer)

            # 2. Middle Segments
            for i in range(1, len(self.intercept_layers)):
                seg_sd = params[self.segment_names[i]]
                h = self.t2j_segments[i](h, cos, sin, state_dict=seg_sd)
                layer_idx = self.intercept_layers[i]
                intermediates[layer_idx] = h
                if modify_fn is not None:
                    h = modify_fn(h, layer_idx)

            # 3. Final Segment
            final_sd = params[self.segment_names[-1]]
            logits = self.t2j_segments[-1](h, cos, sin, state_dict=final_sd)
            return logits, intermediates
