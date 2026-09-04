"""Alias module redirecting to substrate.interception."""

from .interception import (
    InterceptionContext,
    ModifyFn,
    identity_modify,
    run_with_hooks,
)

__all__ = [
    "InterceptionContext",
    "ModifyFn",
    "identity_modify",
    "run_with_hooks"
]