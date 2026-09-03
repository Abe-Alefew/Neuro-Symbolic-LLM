"""Alias module redirecting to substrate.interception."""

from .interception import (
    InterceptionContext,
    ModifyFn,
    identity_modify,
    run_with_hooks,
    run_with_interception,
)

__all__ = [
    "InterceptionContext",
    "ModifyFn",
    "identity_modify",
    "run_with_hooks",
    "run_with_interception",
]
