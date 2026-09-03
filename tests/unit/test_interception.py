"""Unit tests for substrate.interception module.

Tests hidden-state interception, hook registration, pristine state caching,
downstream propagation, and clean hook lifecycle.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import pytest

from substrate.architecture import Architecture
from substrate.interception import (
    InterceptionContext,
    identity_modify,
    run_with_hooks,
    run_with_interception,
)


class MockBlock:
    def __init__(self, name: str, return_tuple: bool = False):
        self.name = name
        self.return_tuple = return_tuple
        self.hooks = []

    def register_forward_hook(self, hook_fn):
        self.hooks.append(hook_fn)
        handle = SimpleNamespace(remove=lambda: self.hooks.remove(hook_fn))
        return handle

    def forward(self, x):
        out = x + 1.0  # simulate layer transform
        for h in self.hooks:
            out = h(self, (x,), out)
        if self.return_tuple:
            return (out, "extra_cache")
        return out


class MockModel:
    config: Any = None

    def __init__(
        self,
        num_layers: int = 4,
        return_tuple: bool = False,
        config: Any = None,
    ):
        self.blocks = [
            MockBlock(f"h_{i}", return_tuple=return_tuple) for i in range(num_layers)
        ]
        self.transformer = SimpleNamespace(h=self.blocks)
        self.config = config


class TestIdentityModify:
    def test_returns_same_object(self):
        obj = [1, 2, 3]
        assert identity_modify(obj, 0) is obj


class TestInterceptionContext:
    def test_registers_and_cleans_up_hooks(self):
        model = MockModel(num_layers=4)
        arch = Architecture(
            model_family="gpt2",
            num_layers=4,
            hidden_size=64,
            num_heads=2,
            head_dim=32,
            vocab_size=100,
        )

        assert len(model.blocks[1].hooks) == 0
        with InterceptionContext(model, arch, intercept_layers=[1, 3]) as ctx:
            assert len(model.blocks[0].hooks) == 0
            assert len(model.blocks[1].hooks) == 1
            assert len(model.blocks[2].hooks) == 0
            assert len(model.blocks[3].hooks) == 1
        # Upon exit, all hooks removed
        assert len(model.blocks[1].hooks) == 0
        assert len(model.blocks[3].hooks) == 0

    def test_cleans_up_on_exception(self):
        model = MockModel(num_layers=4)
        arch = Architecture(
            model_family="gpt2",
            num_layers=4,
            hidden_size=64,
            num_heads=2,
            head_dim=32,
            vocab_size=100,
        )

        with pytest.raises(RuntimeError):
            with InterceptionContext(model, arch, intercept_layers=[1]):
                assert len(model.blocks[1].hooks) == 1
                raise RuntimeError("Simulated forward failure")

        assert len(model.blocks[1].hooks) == 0

    def test_captures_pristine_state_and_modifies(self):
        model = MockModel(num_layers=4, return_tuple=True)
        arch = Architecture(
            model_family="gpt2",
            num_layers=4,
            hidden_size=64,
            num_heads=2,
            head_dim=32,
            vocab_size=100,
        )

        def add_ten(h, layer_idx):
            return h + 10.0

        with InterceptionContext(
            model, arch, intercept_layers=[1], modify_fn=add_ten
        ) as ctx:
            # Simulate block 1 execution
            # input 0.0 -> pristine output 1.0 -> modified output 11.0
            out = model.blocks[1].forward(0.0)

            # Check that return format is preserved (tuple)
            assert isinstance(out, tuple)
            assert out[0] == 11.0
            assert out[1] == "extra_cache"

            # Check that intermediates captured the PRISTINE state (1.0, not 11.0)
            assert 1 in ctx.intermediates
            assert ctx.intermediates[1] == 1.0

    def test_auto_detects_arch_from_model_config(self):
        model = MockModel(num_layers=4)
        model.config = SimpleNamespace(
            model_type="gpt2",
            n_layer=4,
            n_embd=64,
            n_head=2,
            vocab_size=100,
        )
        with InterceptionContext(model, intercept_layers=[1]) as ctx:
            assert ctx.arch.num_layers == 4
            assert len(model.blocks[1].hooks) == 1
        assert len(model.blocks[1].hooks) == 0

    def test_raises_when_no_arch_and_no_config(self):
        model = MockModel(num_layers=4)
        with pytest.raises(ValueError, match="arch must be provided"):
            InterceptionContext(model, intercept_layers=[1])


class TestRunWithInterception:
    def test_functional_loop_interception(self):
        def embed_fn(p, ids):
            return ids * 1.0

        def make_block(i):
            return lambda p, h: h + 1.0

        block_fns = [make_block(i) for i in range(4)]

        def head_fn(p, h):
            return h * 2.0

        def perturb_hook(h, idx):
            return h + 100.0

        # Without interception
        logits_base, inter_base = run_with_interception(
            embed_fn, block_fns, head_fn, params={}, input_ids=0.0
        )
        # 0 -> 1 -> 2 -> 3 -> 4 -> * 2 = 8
        assert logits_base == 8.0
        assert inter_base == {}

        # Intercept at layer 1 (after block 1)
        # block 0: 0 + 1 = 1
        # block 1: 1 + 1 = 2 (pristine h_1 = 2) -> modify: 2 + 100 = 102
        # block 2: 102 + 1 = 103
        # block 3: 103 + 1 = 104
        # head: 104 * 2 = 208
        logits_mod, inter_mod = run_with_interception(
            embed_fn,
            block_fns,
            head_fn,
            params={},
            input_ids=0.0,
            intercept_layers=[1],
            modify_fn=perturb_hook,
        )
        assert logits_mod == 208.0
        assert inter_mod[1] == 2.0  # pristine pre-modification state

    def test_multiple_interception_points(self):
        def embed_fn(p, ids):
            return ids * 1.0

        block_fns = [lambda p, h: h + 1.0 for _ in range(4)]

        def head_fn(p, h):
            return h

        def hook(h, idx):
            return h * 10.0

        # Intercept at layer 0 and layer 2
        # block 0: 0 + 1 = 1 -> pristine 1 -> modified 10
        # block 1: 10 + 1 = 11
        # block 2: 11 + 1 = 12 -> pristine 12 -> modified 120
        # block 3: 120 + 1 = 121
        logits, intermediates = run_with_interception(
            embed_fn,
            block_fns,
            head_fn,
            params={},
            input_ids=0.0,
            intercept_layers=[0, 2],
            modify_fn=hook,
        )
        assert intermediates[0] == 1.0
        assert intermediates[2] == 12.0
        assert logits == 121.0
