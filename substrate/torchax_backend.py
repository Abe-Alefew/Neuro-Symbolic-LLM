"""Device management, dispatch setup, and value-conversion utilities for
running PyTorch code on TorchAX's JAX-backed device.

This module is deliberately separate from `torchax_gpt2.py`: loading and
freezing a specific HF checkpoint is one concern, and getting values on and
off the JAX-backed device correctly is a completely orthogonal one. Anything
that needs to move a value between torch-side and real-JAX-side code --
Person 2's residual, the eventual FabricPC integration, or any future model
loader beyond GPT-2 -- needs what's here without needing `torchax_gpt2.py`
at all.

Three things are provided:

1. `enable_torchax()` / `is_torchax_enabled()` -- idempotent global dispatch
   setup, safe to call from multiple entry points without caring which one
   runs first. TorchAX's own `enable_globally()` is not itself idempotent
   in the sense of being cheap to call repeatedly with no signal of prior
   state; this wraps it so callers never have to coordinate who's
   responsible for enabling it.

2. `to_jax_array()` / `from_jax_array()` -- thin, documented wrappers around
   `torchax.interop.jax_view` / `torch_view`. These are genuine value
   conversions between a `torchax.tensor.Tensor` and a raw `jax.Array` --
   no autograd machinery involved. Use these when a real JAX function needs
   a value that arrived as a torch tensor, or vice versa, and gradient flow
   is not a concern at that particular boundary.

3. `call_jax_differentiable()` -- packages the pattern found by reading
   TorchAX's `interop.py` source directly (not documented in its
   docstrings): `j2t_autograd`'s wrapped function must itself be written in
   torch syntax, because `call_jax`'s argument processing runs `jax_view`
   on the function being wrapped, which assumes torch-syntax semantics.
   A genuine raw-JAX function (real FabricPC node/energy code, no torch
   ops at all) has to be nested one level deeper via `interop.call_jax`
   inside a trivial torch-syntax shell. Verified against pure-torch ground
   truth (forward value and every gradient) in this module's tests --
   this is not a theoretical pattern, it is the one that was empirically
   confirmed to produce correct gradients.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import torch
import torchax
import torchax.interop as interop

T = TypeVar("T")

_torchax_enabled = False


def enable_torchax() -> None:
    """Enable TorchAX's global op interception.

    Idempotent: safe to call from any number of entry points (this module,
    torchax_gpt2.py, Person 2's own code) without needing to coordinate
    who calls it first or worrying about calling it twice.
    """
    global _torchax_enabled
    if not _torchax_enabled:
        torchax.enable_globally()
        _torchax_enabled = True


def is_torchax_enabled() -> bool:
    """Whether `enable_torchax()` has been called in this process yet."""
    return _torchax_enabled


def to_jax_array(value: Any) -> jax.Array:
    """Convert a `torchax.tensor.Tensor` (or plain torch.Tensor moved onto
    the 'jax' device) into a raw `jax.Array`.

    This is a value conversion, not an autograd boundary -- the result is
    a genuine jax.Array with no torch-autograd history attached. Use this
    when handing a value to real JAX/FabricPC code that doesn't need
    gradients to flow back through this specific call (e.g. inspecting a
    value, or as a building block inside `call_jax_differentiable`, which
    handles the gradient-preserving case).
    """
    enable_torchax()
    return interop.jax_view(value)


def from_jax_array(value: jax.Array) -> torch.Tensor:
    """Convert a raw `jax.Array` into a `torchax.tensor.Tensor`.

    Same caveat as `to_jax_array`: a plain value conversion, no autograd
    history. See `call_jax_differentiable` for the gradient-preserving
    version.
    """
    enable_torchax()
    return interop.torch_view(value)


def to_torchax_device(obj: T) -> T:
    """Move a `torch.nn.Module` or `torch.Tensor` onto TorchAX's 'jax'
    device, enabling global dispatch first if it hasn't been already.
    """
    enable_torchax()
    return obj.to("jax")


def is_on_torchax_device(tensor: torch.Tensor) -> bool:
    """Whether `tensor` is already on TorchAX's 'jax' device."""
    return str(tensor.device).startswith("jax")


def call_jax_differentiable(
    jax_fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap a genuine raw-JAX function so it can be called from torch-side
    code (e.g. from inside a `register_forward_hook` callback on a
    TorchAX-backed model) with correct, verified autograd gradients.

    `jax_fn` should be ordinary JAX code -- `jnp`/`jax.lax` operations, or
    a real FabricPC node/energy computation -- with no torch syntax
    anywhere in it. Nothing about `jax_fn` needs to know TorchAX exists.

    Example:
        def fabricpc_residual(h0, W, b):
            return h0 + jnp.tanh(jnp.matmul(h0, W) + b)   # real JAX, unmodified

        bridge = call_jax_differentiable(fabricpc_residual)

        def hook(module, inp, output):
            return bridge(output, W, b)   # W, b: torch tensors on 'jax' device,
                                           # requires_grad=True

    Why this needs a shell rather than wrapping `jax_fn` directly: found by
    reading `torchax/interop.py` source. `j2t_autograd`'s `call_jax`
    argument processing runs `jax_view` on the function argument itself,
    which assumes it is torch-syntax code and wraps it via `call_torch`
    semantics -- so a function containing raw `jnp` calls fails the moment
    it's invoked, because the values arriving in its body are torch-side
    objects, not raw jax arrays. The fix is to nest the real computation
    one level deeper: a trivial torch-syntax shell (satisfies
    `j2t_autograd`'s calling convention) that does nothing but hand off to
    the real function via `interop.call_jax` (a genuine value conversion,
    happening *inside* the same `jax.vjp` trace `j2t_autograd` already
    established, so nothing is severed).
    """
    enable_torchax()

    def torch_shell(*args: Any, **kwargs: Any) -> Any:
        return interop.call_jax(jax_fn, *args, **kwargs)

    return interop.j2t_autograd(torch_shell)