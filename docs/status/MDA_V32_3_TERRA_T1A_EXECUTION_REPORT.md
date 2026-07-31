# Terra T0/T1A execution report

**Execution ID:** `mda-v32-3-artifact-search-20260731-v1`
**Completed:** 2026-07-31T04:56:00Z
**Scope:** T0 and T1A only; HMDD v3.2 / MDAv3.2-3 only

## T0

- Reviewed branch/base: `codex/v32-portable-d1` at
  `9b35346ed282b167c89d713153571afedb6dd46c`.
- The tracked tree was clean. The only untracked path was the pre-existing, untouched
  `~$oCao_DHGCMDA.docx` Word lock file.
- Platform/Python and the seven-file MDAv3.2 inventory are captured in the execution ledger;
  immutable hashes remain the S0 ledger values.
- No `.gitignore` rule was needed: no repository quarantine was created and all search metadata
  is tracked. Any future candidate bytes must remain in an external data root.

## T1A result

All registered steps L1–L5 were executed in order. No data-bearing candidate was retrieved and
no bytes were downloaded, hashed, extracted, or placed in the repository.

The authorized S1B coverage correction added a public Crossref DOI metadata check and screened
all twelve OpenAlex citing works by exact title, DOI, and public landing metadata. Crossref has
only article/PDF links and no explicit supplement/code/repository relation. No citing-work
landing metadata exposed an explicit public code/data link. A single 16-second, no-retry exact-title
GitHub/GitLab/Gitee batch was recorded as blocked when every unresolved request remained incomplete;
the previously found `MHNNMDA` repository is a duplicate source-only, scalar-loader record, not a
new data-bearing candidate.

- L1: the publisher search result confirms the published dimensions and points to the former
  original GitHub URL. That namespace now returns public HTTP 404; Oxford blocked automated
  supplement-link inspection with a Cloudflare challenge.
- L2: public GitHub repository searches returned no target repositories. GitHub code search
  requires authentication; no credentials were used. The absent original namespace gave no
  bounded GH Archive event/object target.
- L3: Software Heritage has no matching origin; Wayback returned no GitHub capture. The
  publisher-page CDX query timed out.
- L4: no relevant public records/projects were returned by Zenodo, DataCite, Figshare, GitLab,
  or Gitee; the OSF query endpoint returned 404. Numeric-token false positives were inspected
  only enough to reject relevance.
- L5: OpenAlex returned 12 citing works. The sole discovered related code repository,
  `CDMBlab/MHNNMDA`, contains source only; its loader names scalar matrix inputs and no target
  data, archive, entity map, or five-type representation.

The complete schema-conformant query/candidate ledger and the three metadata-only/rejected
records are in [MDA_V32_3_TERRA_T1A_EXECUTION_LEDGER_V1.json](MDA_V32_3_TERRA_T1A_EXECUTION_LEDGER_V1.json).
No A/B/C identity decision or semantic ambiguity decision was made.

## Time and handoff

The ledger charges 31 active T1A minutes. With the registered Sol S1A charge of two hours, this
is 2h31m against the combined 8–10 hour S1 cap; the listed stop condition is **search ladder
exhausted**, not budget exhaustion.

Return to Sol S1B. The remaining blockers are the inaccessible/absent primary repository,
unresolved publisher supplement access, and no custody-bearing entity/triple artifact. R2/R3,
P3, reconstruction, training, and history changes remain unauthorized.
