# Sol S1B review — MDAv3.2-3 artifact search v1

**Reviewed:** 2026-07-31
**Scope:** S1B only; HMDD v3.2 / MDAv3.2-3
**Search ID:** `mda-v32-3-artifact-search-20260731-v1`
**Decision:** no exact-artifact candidate found; proceed to the labelled class-B reconstruction path.

## Verdict

The search ledger is frozen. No candidate qualifies for R4 review, so no class-A benchmark claim
is available. This is a search result, not proof that the artifact never existed.

The next authorized scientific/data-contract work is R2/S2: pin inputs and write a declarative
candidate-reconstruction specification. The dataset identity remains `unknown`; there is not yet a
class-B dataset and R4 must still assign the eventual A/B/C class.

## Reviewed evidence

- Terra completed all five registered ladder steps and the later bounded coverage correction.
- The final ledger has 29 query entries: 27 base-ladder entries plus two correction entries;
  it also screens all 12 OpenAlex citing works.
- No candidate bytes were downloaded, extracted, or placed in the repository. Therefore no
  candidate can pass the hash, parser, entity, five-type, or typed-triple acceptance tests.
- C001 is publisher article metadata only; it establishes the published claim but not entity or
  triple identity.
- C002 is the former original repository reference; its current public GitHub namespace returns
  HTTP 404, which does not prove historical absence.
- C003 (`CDMBlab/MHNNMDA`) is a source-only citing implementation. Its inspected tree contains
  no data/archive/entity-map object, and its loader names scalar matrix inputs; it is correctly
  rejected for class A.
- The DOI metadata record exposes article/PDF links only. It has no explicit supplement, code,
  repository, archive, or relation link.

## Counterevidence and caveats

- Oxford blocked automated supplement inspection with Cloudflare.
- GitHub code search required authentication; no credentials were used.
- Software Heritage had no matching origin; Wayback had no captured GitHub page, while the
  publisher CDX request timed out.
- The title-forge batch for the 12 citing works was bounded and recorded as blocked after its
  16-second no-retry limit. No public landing metadata exposed a code/data link.

These limitations mean the ledger supports “no artifact found under the registered public
ladder,” not a claim of universal absence. Reopening the search requires a newly identified,
in-scope immutable URL, archive identifier, or provenance-bearing lead; repeating blocked
endpoints alone is not authorized.

## Acceptance decision

| Candidate | S1B disposition | Reason |
|---|---|---|
| C001 — publisher metadata | metadata only | No downloadable artifact, entity mapping, or typed triples. |
| C002 — former original GitHub URL | metadata only | Current 404; no retrievable object or custody chain. |
| C003 — MHNNMDA source repository | rejected for class A | No data-bearing object, entity map, or five-type representation. |

No candidate is `r4_ready`. No A/B/C dataset identity was assigned.

## Frozen artifacts and accounting

- [Terra execution ledger](MDA_V32_3_TERRA_T1A_EXECUTION_LEDGER_V1.json)
- [Terra execution report](MDA_V32_3_TERRA_T1A_EXECUTION_REPORT.md)
- [Artifact acceptance contract](MDA_V32_3_ARTIFACT_ACCEPTANCE_V1.md)
- [Registered search ladder](MDA_V32_3_SEARCH_LADDER_V1.json)

S1 accounting: Sol S1A charged 120 minutes; Terra T1A charged 31 minutes; combined total is
151 minutes (2h31m), below the 8–10 active-hour cap. The valid stop condition is registered
ladder exhaustion, not budget exhaustion.

## Next authorized Terra task

Do not resume T1A. Await a Sol S2 reconstruction specification. The eventual next Terra wave is
R2/T1B source-manifest and fetch/verify implementation, followed by R3/T2 only after that
specification is reviewed.
