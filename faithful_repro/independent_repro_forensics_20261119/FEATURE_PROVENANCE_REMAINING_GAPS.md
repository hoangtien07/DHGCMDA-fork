# FEATURE_PROVENANCE_REMAINING_GAPS

1. **Bundle views (sim_m1/sim_m2/sim_d1/sim_d2)**: UNKNOWN_PROVENANCE. We verified they are not trivial full-A GIP (a1_arm canary), but cannot prove no association-derived content (e.g. MISIM-style construction upstream uses associations). Until recipe-traced, results on bundle views are marked "benchmark-bundle", never "clean independent".
2. **PUBLIC_RECONSTRUCTED_4VIEW** exists only partially: seq (NW) + semantic (MeSH-2019) + gene (CTD/HumanNet) have hashed sources in ext/seqgene; the functional view relies on MISIM d2025.bin whose own construction is association-derived upstream → treat as INDEPENDENT_EXTERNAL-for-input but note biological dependency.
3. Paper's exact preprocessing (normalization, KNN parameters inside views, date/version of miRBase/MeSH/CTD) not documented → recipe-level equivalence unattainable without author artifact.
4. Action needed to close: none possible without author release (owner forbids contact) → gap is PERMANENT under current constraints; recorded, not hidden.
