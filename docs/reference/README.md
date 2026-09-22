# docs/reference — extracted formula/parameter sheets

Short verified sheets extracted from papers, so the agent reads a one-page sheet
instead of a PDF (see `docs/03-references.md`). PDFs live in `papers/` (gitignored)
for humans only.

## Provenance rule (enforced)

Every sheet carries a header:

```
Source:       <paper>, <venue year>, <section/table/equation>
Version:      <exact source version pinned — e.g. arXiv:2503.18114v2,
              bioRxiv 10.1101/2024.02.26.582157 v2>. Silent version drift is a
              known failure mode; this field is mandatory.
Derived from: <what was actually read to produce this — e.g. "arXiv HTML table",
              "PDF text", or another repo file>. MUST NOT be another doc in this
              repo (that creates circular verification — see F2, 2026-07-31).
Transcribed by: <agent|human>, <date>
Verified by (human): ____________          # blank until checked
Status: verified | inferred | unverified
```

`Source` is the citation of record; `Derived from` is what was physically
consulted. They differ exactly when the risk lives: a cell "transcribed by agent"
whose `Derived from` is another repo file is **not** independently verified, no
matter how confident it looks.

**A sheet whose `Verified by (human)` is blank MUST NOT be cited by code.**
"If a number appears in `src/`, it appears here with a source" (`03-references.md`).

## Status

| Sheet | Owner | State |
|---|---|---|
| `validation-settings.md` | agent-drafted → human | **signed 2026-09-01** (Chou B.5) |
| `manifold-generator.md` | agent-drafted → human | **signed 2026-09-01** (Chou D.1.1; labels are a stated deviation) |
| `parameterization.md` | agent-drafted → human | **Table 1 + V1 signed 2026-09-01**; A.3 dangling |
| `cl-metrics.md` | agent-drafted → human | A.1–A.10 signed 2026-09-01; running max; §A.7 |
| `glue-algorithm.md` | **human** (Algorithm 2 + Def B.6; high transcription risk) | not written |
| `glue-decomposition.md` | agent-drafted (ICLR 2026 §B.3, exact 3-factor identity + a/b/c) | awaiting human verify |
| `glue-core-validation.md` | agent-**measured** | **signed 2026-09-01** (B.5 numbers spotted against JSON; §5a present) |
| `glue-sign-conventions.md` | **human** (App B.4; ρ_a vs ρ_c opposite, ψ non-monotone) | not written |
| `correlation-duality.md` | agent-draft → human verify vs PRL supplement (+ Gaussianization, `00` §6.3) | not written |
| `glue-refinements.md` | agent-draft (main text) → human verify + extract S1 | drafted; which-assumptions done, A2↔C4 cross-ref added, S1 formula pending |
| `parameterization-derivation.md` | independent derivation, V1 signed | A.3 dangling; `(N/N_base)` confirmed |
| `protocol-deviations.md` | human (D.1.1 balanced-dichotomy departure) | **written 2026-09-01** |
| `generalization-metrics.md` | human (Johnston & Fusi probe metric) | not written |
| `ccgp-protocol.md` | agent-draft → human verify (Bernardi 2020) | not written |
| `optimal-coding-statistics.md` | **human** (Wakhloo/Slatton four statistics; high risk) | not written |

The human-owned sheets are the ones where a transcription error produces
plausible wrong *magnitudes* that the `02` §1 sign tests will not catch.
