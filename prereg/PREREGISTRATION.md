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

### 2026-10-02 — Query emission rate

Before reader training or condition evaluation, increase the interleaved query
emission rate from Poisson(λ=1.5) to Poisson(λ=3.0). With 2,000 episodes this
meets the pre-specified ≥5,000-query evaluation-set size without adding
non-Poisson floor-fill emissions.

### 2026-10-02 — Day 1 generator review

Before reader training or condition evaluation, require within-episode fact-key
uniqueness; give OWNS a distinct J-token vocabulary; tighten type B and C
provenance to their causal episodes; and define type R cues as unique
partner/place/time triples with a logged earlier-competitor count. Conflicts are
paired WORKS→MOVED superseding events in a strictly later time bucket.

### 2026-10-02 — Restatement control

Before reader training or condition evaluation, add `p_restate = 0.3` as the
maximum realized fraction of sampled WORKS/MOVED events that exactly repeat an
already-seen `(entity, relation, object)` fact. Log realized WORKS, MOVED, and
combined restatement rates per world.

### 2026-10-02 — Ground-truth labels and lossless residuals

Before reader training, restrict value labels to facts present in world ground
truth; absent `(entity, relation)` pairs are type-E ABSTAIN. Context-dependent
reader targets use the world value only when explicit or predictive evidence
is present. Residual coding retains every fact and substitutes `PRED` for
canonical prior-predicted values, with explicit EP/RES/GIST/SUM headers and
subject carry-forward in gist traces.

### 2026-10-02 — Exclude type D aggregation

D excluded: oracle evidence ~15k tokens, outside reader context; aggregation is
out of scope for Stage 0.

### 2026-10-02 — Stage 0 demand coverage

Raise the superseding-conflict entity fraction to 30%. Floor each Stage 0
query type's aggregate demand share at 8% after each Dirichlet draw, and reserve
unique supported forms in eval worlds. Calibration samples unique forms without
replacement and fails if a type has fewer than its required sample count.

### 2026-10-02 — R uniqueness floor and H3 subgroup

Set the R uniqueness floor to 250 at all `p_redundancy` levels. Reason:
full-cue uniqueness is scarcer at high redundancy; 275 observed at p=0.9.
Log `from_superseding` on each R query so reinstatement results can be separated
by superseding-event provenance. H3 must also hold on R queries with
`from_superseding = False`.

### 2026-10-02 — Oracle-reader pre-check

Before any reader training, all conditions are evaluated with a symbolic
optimal reader. If the K* ratio of progressive (best decay rule, tuned on
policy-tune worlds) versus the best of `raw_offload` and `raw_offload_qrender`
is < 1.2× at every `p_redundancy` level and every K, Stage 0 stops: the mechanism
lacks information-theoretic headroom. If ≥ 1.5× at any level, proceed to reader
training. Between 1.2× and 1.5×, proceed with reduced priority.
