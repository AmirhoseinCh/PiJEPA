#!/usr/bin/env python3
"""Prove the memoised attention mask is bit-identical to the original loop.

Compares BlockTransformerPt.generate_attention_mask (now layout-keyed memoised)
against a standalone reference that reproduces the original element-by-element
O(total_tokens^2) computation. Runs on CPU; no data or GPU needed.
"""

import numpy as np
import torch

from octo.model.components.block_transformer import AttentionRule
from octo.model.components.block_transformer_pt import (
    BlockTransformerPt,
    PrefixGroupPt,
    TimestepGroupPt,
    TokenMetadataPt,
)


def _mk_prefix(name, n_tokens, rules, batch=2, d=4):
    return PrefixGroupPt(
        tokens=torch.zeros(batch, n_tokens, d),
        mask=torch.ones(batch, n_tokens, dtype=torch.bool),  # all-valid -> combined == structural
        name=name,
        attention_rules=rules,
    )


def _mk_timestep(name, n_tokens, rules, batch=2, horizon=3, d=4):
    return TimestepGroupPt(
        tokens=torch.zeros(batch, horizon, n_tokens, d),
        mask=torch.ones(batch, horizon, n_tokens, dtype=torch.bool),
        name=name,
        attention_rules=rules,
    )


def reference_structural_mask(prefix_groups, timestep_groups, side):
    """Verbatim reproduction of the original loop (returns the 2D structural mask)."""
    horizon = timestep_groups[0].tokens.shape[1]
    tokens_per_prefix_group = [g.tokens.shape[1] for g in prefix_groups]
    tokens_per_timestep_group = [g.tokens.shape[2] for g in timestep_groups]
    tokens_for_prefix = sum(tokens_per_prefix_group)
    tokens_per_time_step = sum(tokens_per_timestep_group)
    total_tokens = tokens_for_prefix + tokens_per_time_step * horizon
    mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool)

    def _get_position(i, tokens_per_elem):
        return np.searchsorted(np.cumsum(tokens_per_elem), i, side=side)

    def meta(i):
        if i < tokens_for_prefix:
            return TokenMetadataPt.create(prefix_groups[_get_position(i, tokens_per_prefix_group)], -1)
        i -= tokens_for_prefix
        t, i = divmod(i, tokens_per_time_step)
        return TokenMetadataPt.create(timestep_groups[_get_position(i, tokens_per_timestep_group)], t)

    for i in range(total_tokens):
        for j in range(total_tokens):
            mask[i, j] = int(meta(i).should_attend_to(meta(j)))
    return mask


def _fresh_block_transformer(use_correct_attention=True):
    """Instantiate without building the heavy TransformerPt submodule."""
    bt = BlockTransformerPt.__new__(BlockTransformerPt)
    bt.enforce_causal = False  # skips verify_causality; irrelevant to mask values
    bt.use_correct_attention = use_correct_attention
    bt._attention_mask_cache = {}
    return bt


# A few layouts that exercise every AttentionRule branch, mimicking Octo's
# task-prefix + obs/readout-timestep structure.
LAYOUTS = [
    dict(
        prefix=[("task_lang", 3, {"*": AttentionRule.ALL})],
        timestep=[
            ("obs", 5, {"task*": AttentionRule.ALL, "obs": AttentionRule.CAUSAL, "readout*": AttentionRule.NEVER}),
            ("readout_action", 1, {"task*": AttentionRule.ALL, "obs": AttentionRule.CAUSAL, "readout*": AttentionRule.CURRENT}),
        ],
        horizon=3,
    ),
    dict(  # different sizes + horizon -> must get its own cache entry
        prefix=[("task_lang", 2, {"*": AttentionRule.ALL}), ("task_img", 4, {"obs": AttentionRule.STRICT_PAST, "*": AttentionRule.ALL})],
        timestep=[("obs", 7, {"task*": AttentionRule.ALL, "obs": AttentionRule.CURRENT})],
        horizon=4,
    ),
]


def _build(layout):
    prefix = [_mk_prefix(n, k, r) for (n, k, r) in layout["prefix"]]
    ts = [_mk_timestep(n, k, r, horizon=layout["horizon"]) for (n, k, r) in layout["timestep"]]
    return prefix, ts


def test_identical_to_reference():
    for side_correct in (True, False):
        side = "right" if side_correct else "left"
        for layout in LAYOUTS:
            prefix, ts = _build(layout)
            bt = _fresh_block_transformer(use_correct_attention=side_correct)
            got = bt.generate_attention_mask(prefix, ts)          # (batch, 1, T, T)
            ref = reference_structural_mask(prefix, ts, side)     # (T, T)
            # All group masks are all-ones, so the padding mask is all-ones and
            # the combined mask equals the structural mask for every batch/head.
            assert got[0, 0].shape == ref.shape, (got.shape, ref.shape)
            assert torch.equal(got[0, 0], ref), f"mismatch for side={side}, layout={layout['prefix']}"
            assert torch.equal(got[0, 0], got[-1, 0]), "structural mask must not vary across batch"
    print("PASS: memoised mask bit-identical to reference across layouts/sides")


def test_cache_hit_is_consistent_and_keyed():
    prefix, ts = _build(LAYOUTS[0])
    bt = _fresh_block_transformer()
    a = bt.generate_attention_mask(prefix, ts)
    b = bt.generate_attention_mask(prefix, ts)  # cache hit
    assert torch.equal(a, b)
    assert len(bt._attention_mask_cache) == 1
    # A different layout must produce a distinct, separately-cached mask.
    prefix2, ts2 = _build(LAYOUTS[1])
    c = bt.generate_attention_mask(prefix2, ts2)
    assert len(bt._attention_mask_cache) == 2
    assert c.shape[-1] != a.shape[-1]
    print("PASS: cache is layout-keyed and stable on hits")


if __name__ == "__main__":
    test_identical_to_reference()
    test_cache_hit_is_consistent_and_keyed()
    print("ALL TESTS PASSED")
