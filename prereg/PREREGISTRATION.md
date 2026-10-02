# Aftertrace Stage 0 — Pre-registration

Commit this file before any main-run results exist. Amendments are appended
below with a date and reason; the original text is never edited.

## Hypotheses

H1 (primary). At matched context tokens per query, `progressive` (best
decay rule, chosen on policy-tuning worlds) achieves higher utility than
both `raw_offload` and `residual_offload`, and the K* ratio at 90% of
ceiling is ≥ 1.5×.

H2. The H1 advantage grows as K shrinks: the ratio at K ≤ 64 exceeds the
ratio at K = 256.

H3 (reinstatement). For latent items, retrieval correctness increases with
cue diagnosticity (candidate-set size 8 → 1) while confabulation rate does
not increase.

## Reader calibration (must pass before any condition comparison)

Select the smallest reader size meeting all of:
1. `full_oracle` utility within 2 points of r31m.
2. Between-seed SD < 1/3 of the minimum effect of interest (3 points).
3. `none` within 5 points of chance.
4. Criterion 1 also holds on query types B and C.
Run main experiments at the selected size and the next size up.

## Minimum effect of interest

3 points of utility at matched K, or 1.5× K* ratio.

## Kill the hypothesis if (at a calibrated reader, at both sizes)

- `raw_offload` or `residual_offload` is within CI of `progressive` at every K;
- the advantage at K ≤ 64 is not larger than at K = 256;
- the advantage disappears after the demand shift (second half of stream);
- `summary` matches `progressive` within CI;
- confabulation rises with cue diagnosticity in Fig 2.

## Kill only the implementation (fix and rerun) if

- calibration fails → scale the reader;
- seed SD exceeds the minimum effect → fix training;
- `none` is far above chance → leakage bug.

## Scoping outcome (not a kill)

Advantage present only at p_redundancy = 0.9 → report as "helps when the
prior is strong."

## Continue thresholds

- Another weekend: Fig 1 gap ≥ 2 points at K = 32, right direction.
- One month: K* ratio ≥ 1.5× (CI excludes 1) at both sizes, survives demand shift.
- Cloud GPU / natural-language stage: ratio ≥ 2× and clean Fig 2.

## External judge

Before calling any result "promising," send Fig 1, Fig 2, and the
accounting table to: ______________________ (fill in before the main run).

## Amendments

(none)
