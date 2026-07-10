"""JetSpec draft head — causal parallel tree drafting.

JetSpec shares DFlash's architecture exactly (Qwen3-style decoder layers with frozen-target
hidden states injected as contextual K/V, projected via a bias-free `fc` + RMSNorm). The one
structural difference is the *within-block* attention: JetSpec makes it **causal** (each draft
position attends only to the prefix + earlier positions in its block) instead of DFlash's
bidirectional block. That restores the target's autoregressive factorization per depth, so the
per-depth marginals form coherent speculation trees.

`JetSpecDraftModel` is therefore a thin specialization of `DFlashDraftModel` that forces the
causal head on, so a config with ``architectures: ["JetSpecDraftModel"]`` (or ``head_type:
"causal"``) trains/loads a JetSpec head without any extra flag.

- Train:  ``scripts/train_jetspec.py`` (or ``scripts/train_dflash.py --head-type causal``).
- Tree-decode inference / acceptance eval: :mod:`specforge.modeling.draft.jetspec_tree`
  and ``scripts/eval_accept_hf.py --mode tree``.
"""
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.jetspec_tree import (
    DraftTree,
    build_accum_logp_tree,
    build_tree_verify_mask,
    tree_spec_generate,
)

__all__ = [
    "JetSpecDraftModel",
    "DraftTree",
    "build_accum_logp_tree",
    "build_tree_verify_mask",
    "tree_spec_generate",
]


class JetSpecDraftModel(DFlashDraftModel):
    """DFlash head with the causal (JetSpec) within-block attention forced on.

    Identical weights/architecture to ``DFlashDraftModel``; only the attention mask pattern
    differs (causal within block). Kept as a distinct class so JetSpec is a first-class,
    named draft type in configs and the model registry.
    """

    def __init__(self, config) -> None:
        # Force the causal head regardless of how the config was authored.
        dflash_cfg = getattr(config, "dflash_config", None) or {}
        if getattr(config, "head_type", None) != "causal" and not dflash_cfg.get("causal_head"):
            config.head_type = "causal"
        super().__init__(config)
        assert self.causal_head, "JetSpecDraftModel requires a causal head (head_type='causal')"
