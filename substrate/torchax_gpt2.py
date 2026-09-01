"""Official HF GPT-2, running correctly under TorchAX, with parameters
exposed as an explicit functional argument.

This module is deliberately narrow in scope: get the real, unmodified
`transformers.GPT2LMHeadModel` executing on JAX (via TorchAX's op dispatch)
with verified numerical fidelity, and expose it in a form Person 2's
interception work can build on. It does not do any interception itself --
that is `Block 0-6 -> INTERCEPT -> Block 7-11 -> LM head` from the shared
architecture diagram, and lives elsewhere.

Design note on the interface, worth keeping explicit: `functional_gpt2`
takes the *model object* as its first argument, not just its params. That
is deliberate. Hooks registered on the model's submodules (e.g.
`model.transformer.h[6].register_forward_hook(...)`) fire correctly even
when the model is subsequently invoked through `torch.func.functional_call`
-- verified directly, not assumed -- which is what lets Person 2 intercept
mid-forward-pass without this module needing to know anything about where
the interception happens. The seam is the hook mechanism, not a parameter
this function has to expose.

Everything here is verified against a locally-constructed random-init
GPT2Config, since this sandbox has no route to huggingface.co (only
pypi.org/github.com are reachable). The one thing that genuinely needs
re-confirming against a real downloaded checkpoint, in an environment that
can reach the Hub, is `load_torchax_gpt2` itself -- the dispatch-fidelity
and freeze mechanics below do not depend on which weight values are loaded,
only on TorchAX's op dispatch behaving consistently, which is what was
actually tested.
"""

from __future__ import annotations

from typing import Any

import torch
import torchax
from torch.func import functional_call

from .provenance import CheckpointProvenance, resolve_checkpoint_provenance

_SUPPORTED_MODEL_TYPES = {"gpt2"}

_torchax_enabled = False


def _ensure_torchax_enabled() -> None:
    """TorchAX's global op interception is a one-time process-wide switch.
    Idempotent on purpose: every entry point in this module calls this
    rather than assuming some other code path already did.
    """
    global _torchax_enabled
    if not _torchax_enabled:
        torchax.enable_globally()
        _torchax_enabled = True


def load_torchax_gpt2(
    model_id: str,
    revision: str | None = None,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], CheckpointProvenance]:
    """Load a real HF GPT-2 checkpoint, moved onto TorchAX's JAX-backed
    device, with its parameters frozen and exposed as an explicit dict.

    Returns:
        model: the live nn.Module, on the 'jax' device. Person 2 registers
            interception hooks on this object's submodules (e.g.
            `model.transformer.h[6]`) *before* calling `functional_gpt2` --
            hook registration is a property of the module, not of any one
            call, so this can happen at any point before the first forward.
        params: dict of parameter name -> tensor (on the 'jax' device),
            every leaf with requires_grad=False. This is the explicit
            functional argument `functional_gpt2` and downstream code
            (jax.jit, a training step, etc.) should thread through -- never
            read parameters off `model` directly once this is built, or
            the "explicit functional argument" property is just cosmetic.
        provenance: pinned to a resolved commit SHA (see
            substrate.provenance) -- same guarantee this project already
            relies on elsewhere, extended to this loading path rather than
            reimplemented for it.

    Raises:
        ValueError: if the checkpoint's architecture family isn't
            supported yet (currently gpt2 only; gpt_neox/Pythia support is
            a deliberate follow-up, not an oversight -- see module
            docstring in substrate.architecture for why family-specific
            behavior should stay isolated rather than creep in here).
    """
    _ensure_torchax_enabled()

    from transformers import AutoConfig, AutoModelForCausalLM  # local import: heavy dep

    provenance = resolve_checkpoint_provenance(model_id, revision)
    pinned_revision = provenance.resolved_sha or provenance.requested_revision

    config = AutoConfig.from_pretrained(model_id, revision=pinned_revision)
    if config.model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model architecture {config.model_type!r} for model "
            f"{model_id!r}. Supported: {sorted(_SUPPORTED_MODEL_TYPES)}."
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=pinned_revision)
    model.eval()
    model = model.to("jax")

    params = dict(model.named_parameters())
    for p in params.values():
        p.requires_grad_(False)

    return model, params, provenance


def functional_gpt2(
    model: torch.nn.Module,
    params: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
) -> Any:
    """Run `model`'s forward pass with `params` as an explicit argument
    rather than the module's own internal state.

    This is a thin wrapper around `torch.func.functional_call` -- the
    thinness is deliberate. Any hooks already registered on `model`'s
    submodules fire normally during this call (verified directly against
    this exact call path, not assumed from functional_call's docs); this
    function does not need to know they exist.

    `input_ids` must already be on the same device as `params` (i.e.
    `.to("jax")` if `params` came from `load_torchax_gpt2`).
    """
    return functional_call(model, params, (input_ids,))


def check_numerical_fidelity(
    model_id: str,
    revision: str | None = None,
    seq_len: int = 8,
    batch_size: int = 2,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Compare the TorchAX-dispatched forward pass against a plain,
    non-TorchAX PyTorch forward pass on the *same* loaded weights and the
    same random input.

    This is the check that actually matters for this project: TorchAX
    "running without error" says nothing about whether its op dispatch
    reproduces real PyTorch numerics. Verified directly on a random-init
    GPT2Config in this module's own tests, at ~9e-8 max abs diff -- the
    same order of magnitude fidelity this project already holds itself to
    elsewhere (models.py vs. real torch). Re-run this against any real
    downloaded checkpoint before trusting it for that checkpoint
    specifically; dispatch fidelity for one set of weights does not
    logically guarantee it for another, even if there is no concrete
    reason to expect it to differ.
    """
    import copy

    from transformers import AutoConfig, AutoModelForCausalLM

    _ensure_torchax_enabled()

    provenance = resolve_checkpoint_provenance(model_id, revision)
    pinned_revision = provenance.resolved_sha or provenance.requested_revision
    config = AutoConfig.from_pretrained(model_id, revision=pinned_revision)
    model_plain = AutoModelForCausalLM.from_pretrained(model_id, revision=pinned_revision)
    model_plain.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        ref_logits = model_plain(input_ids=ids).logits

    model_jax = copy.deepcopy(model_plain).to("jax")
    params = dict(model_jax.named_parameters())
    for p in params.values():
        p.requires_grad_(False)

    out = functional_gpt2(model_jax, params, ids.to("jax"))
    jax_logits = out.logits.to("cpu")

    diff = (ref_logits - jax_logits).abs()
    return {
        "shapes_match": ref_logits.shape == jax_logits.shape,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose": bool(torch.allclose(ref_logits, jax_logits, atol=atol)),
        "atol": atol,
        "provenance": provenance.as_dict(),
    }