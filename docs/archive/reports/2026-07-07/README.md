# Legacy report bundle from `ca489cb`

This directory is an immutable reference bundle imported from commit `ca489cb`
(`breakthrough-conformal`). It is intentionally namespaced so it cannot be mistaken for
the active report pipeline.

Included:

- `BaoCao_DHGCMDA_HoanThien.docx` — polished Vietnamese report.
- `Seminar_DHGCMDA.docx` — seminar handout.
- `generate_report_v2.py`, `generate_seminar.py`, and the figures used by those documents.

`MANIFEST.sha256` records the SHA-256 values after import; it protects this reference bundle
from accidental edits.

The bundle predates the leakage audit (P7/P9/P10), the Tissue training-label mapping fix,
and Plan N readiness work. Its `0.697` v2.0 headline and claims that rely on it are
legacy/leaky results, not the current honest benchmark. Do not regenerate or publish these
documents without a report-refresh task.
