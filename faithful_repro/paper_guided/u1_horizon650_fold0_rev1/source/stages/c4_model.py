# -*- coding: utf-8 -*-
"""PaperGuidedC4 — frozen C2 encoder + optional ReconstructionAdapter.

Scope per C4_CONTRACT.md: registers the ACCEPTED
ReconstructionAdapter AFTER the unchanged PaperGuidedC2 constructor.
recon_enabled=False -> adapter.heads is None: no parameters, no RNG
draw, identical parameter/state layout to PaperGuidedC2 (bitwise
C3/S1 equivalence preserved). recon_enabled=True -> the accepted
seed+4001 fork_rng factory draws INSIDE fork_rng, so the baseline
parameters and the caller's global RNG stay byte-identical to C2;
only the four decoder weights are appended. The runner must build
the optimizer AFTER this constructor so all four decoder weights
participate. No duplicate module aliases, no new baseline init, no
extra encoder forward — the adapter consumes out['zm_pre']/out
['zd_pre'] from the SAME forward.
"""
import c2_model as c2m
from reconstruction_adapter import ReconstructionAdapter


class PaperGuidedC4(c2m.PaperGuidedC2):
    """C2 model + optional reconstruction adapter (same forward)."""

    def __init__(self, view_dims, *args, recon_enabled=False,
                 recon_seed=0, hidden=256, proj=64, **kwargs):
        super().__init__(view_dims, *args, hidden=hidden, proj=proj,
                         **kwargs)
        # enabled=False: heads=None — no parameters, no RNG, forward
        # returns None (accepted disabled-safe adapter semantics).
        self.recon = ReconstructionAdapter(hidden, proj,
                                           int(recon_seed),
                                           enabled=recon_enabled)
