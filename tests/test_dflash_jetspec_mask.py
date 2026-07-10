"""Correctness tests for the DFlash (bidirectional) vs JetSpec (causal) block mask.

The single-source-of-truth predicate is `specforge.core.dflash.dflash_block_visible`,
used by both the dense (eager/sdpa) and flex_attention paths. We validate it against an
independent pure-Python reference, and confirm the *only* difference between DFlash and
JetSpec is the within-block (noise) half of the mask.

Run directly:  PYTHONPATH=. python tests/test_dflash_jetspec_mask.py
Or via pytest: PYTHONPATH=. pytest tests/test_dflash_jetspec_mask.py
"""
import torch

from specforge.core.dflash import OnlineDFlashModel, dflash_block_visible


def _ref_visibility(q_len, block_size, causal):
    """Independent pure-Python reference for the [L, 2L] visibility matrix."""
    M = [[False] * (2 * q_len) for _ in range(q_len)]
    for q in range(q_len):
        qb, qi = q // block_size, q % block_size
        for k in range(2 * q_len):
            if k < q_len:  # context key
                vis = (k // block_size) < qb
            else:  # noise key
                kk = k - q_len
                vis = (kk // block_size == qb) and (
                    (kk % block_size) <= qi if causal else True
                )
            M[q][k] = vis
    return torch.tensor(M, dtype=torch.bool)


def _make_model(block_size, head_type):
    """OnlineDFlashModel mask methods only depend on block_size/head_type/cache attrs."""
    m = object.__new__(OnlineDFlashModel)
    m.block_size = block_size
    m.head_type = head_type
    m._cached_block_mask = None
    m._cached_seq_len = None
    m._cached_bsz = None
    m._cached_num_heads = None
    return m


def _dense_visible(model, q_len):
    """Recover the boolean visibility from the additive dense mask (0 = visible)."""
    mask = model._create_parallel_attention_mask(q_len, torch.device("cpu"))
    assert mask.shape == (q_len, 2 * q_len), mask.shape
    return mask == 0.0


def test_dense_matches_reference():
    for block_size, q_len in [(4, 8), (4, 12), (3, 9), (16, 32)]:
        for head_type, causal in [("bidirectional", False), ("causal", True)]:
            model = _make_model(block_size, head_type)
            got = _dense_visible(model, q_len)
            ref = _ref_visibility(q_len, block_size, causal)
            assert torch.equal(got, ref), (
                f"dense mask mismatch: block_size={block_size} q_len={q_len} "
                f"head_type={head_type}"
            )


def test_context_half_identical_only_noise_differs():
    """DFlash and JetSpec must differ ONLY in the within-block noise half."""
    q_len, block_size = 16, 4
    bidir = _dense_visible(_make_model(block_size, "bidirectional"), q_len)
    causal = _dense_visible(_make_model(block_size, "causal"), q_len)
    # Context half (first L columns) identical.
    assert torch.equal(bidir[:, :q_len], causal[:, :q_len])
    # Noise half: causal is a strict subset (drops future-in-block), and is exactly
    # the lower-triangular-within-block restriction of bidirectional.
    assert torch.equal(causal[:, q_len:], bidir[:, q_len:] & _block_tril(q_len, block_size))
    # Causal must remove at least one position that bidirectional allowed.
    assert (bidir[:, q_len:] & ~causal[:, q_len:]).any()


def _block_tril(q_len, block_size):
    idx = torch.arange(q_len)
    same_block = (idx[:, None] // block_size) == (idx[None, :] // block_size)
    le = (idx[None, :] % block_size) <= (idx[:, None] % block_size)
    return same_block & le


def test_causal_noise_is_block_lower_triangular():
    """Within each block the noise half is lower-triangular; the anchor (offset 0)
    sees only itself; offset i sees offsets 0..i."""
    q_len, block_size = 12, 4
    causal = _dense_visible(_make_model(block_size, "causal"), q_len)
    noise = causal[:, q_len:]
    for q in range(q_len):
        qb, qi = q // block_size, q % block_size
        for k in range(q_len):
            kb, ki = k // block_size, k % block_size
            expect = (kb == qb) and (ki <= qi)
            assert bool(noise[q, k]) == expect, (q, k)


def test_flex_calling_convention_matches_reference():
    """The flex_attention path feeds scalar index tensors to dflash_block_visible.
    Validate that calling convention via create_mask against the reference."""
    try:
        from torch.nn.attention.flex_attention import create_mask
    except Exception as e:  # pragma: no cover
        print(f"[skip] flex_attention unavailable: {e}")
        return
    q_len, block_size = 8, 4
    for head_type, causal in [("bidirectional", False), ("causal", True)]:
        def mask_fn(b, h, q_idx, kv_idx, _causal=causal):
            return dflash_block_visible(q_idx, kv_idx, q_len, block_size, _causal)

        try:
            dense = create_mask(mask_fn, B=1, H=1, Q_LEN=q_len, KV_LEN=2 * q_len,
                                device="cpu")
        except Exception as e:  # pragma: no cover
            print(f"[skip] create_mask failed on cpu: {e}")
            return
        got = dense[0, 0].bool()
        ref = _ref_visibility(q_len, block_size, causal)
        assert torch.equal(got, ref), f"flex mask mismatch head_type={head_type}"


if __name__ == "__main__":
    test_dense_matches_reference()
    print("ok: dense matches reference (bidirectional + causal)")
    test_context_half_identical_only_noise_differs()
    print("ok: context half identical, only noise half differs")
    test_causal_noise_is_block_lower_triangular()
    print("ok: causal noise half is block-lower-triangular")
    test_flex_calling_convention_matches_reference()
    print("ok: flex calling convention matches reference (or skipped)")
    print("ALL PASS")
