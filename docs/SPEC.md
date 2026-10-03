# Aftertrace Stage 0 — Specification (v0.1)

## 0. Question and currency

Primary currency: **tokens placed in the reader's context per query (K)**.
Secondary currencies: persistent store size (tokens and bits), retrieval
operations per query, maintenance operations per step.

Bytes are deliberately not primary: symbolic episodes are tiny, so raw
storage is nearly free and would trivially favor raw offload. What is
expensive in real systems is context injected per query.

## 1. Repository layout

```
aftertrace/
  world/       vocab.py grammar.py generator.py queries.py seeds.py
  reader/      model.py data.py train.py
  memory/      retriever.py render.py base.py conditions/*.py
  accounting.py
  eval/        harness.py stats.py figures.py
configs/       world.yaml reader_*.yaml conditions.yaml sweep.yaml
prereg/        PREREGISTRATION.md
tests/
runs/          (gitignored)
```

## 2. Vocabulary (symbolic, fixed, < 1,000 tokens)

- Special: PAD BOS EOS SEP EP EPP RES SUM Q A ABSTAIN MASK PRED GIST
- Relations (12): WORKS MOVED JOINED_YEAR TOLD AT MET OWNS COLOR LIVES
  STUDIED MANAGES SIBLING
- Query operators: CURRENT PREV HOP2 CTX
- Values: cities C_0..C_59, companies O_0..O_39, years Y_0..Y_29,
  colors K_0..K_11, places P_0..P_19, owned objects J_0..J_15
- Entities: E_0..E_511 (abstract slots, remapped per world/sequence)
- Time buckets: T_0..T_63
- Filler (incidental noise): F_0..F_63

## 3. World generator

A world is a pure function of `(seed, config)`.

**Grammar prior (shared across all worlds — this is the "semantic
knowledge").** Each company has a home city; each city has a typical place
set; each company has a hiring-year window. Parameter `p_redundancy`
controls how often facts follow the prior:
- MOVED city = company's home city with prob `p_redundancy`, else uniform.
- JOINED_YEAR uses the canonical first year in the company window with prob
  `p_redundancy`, otherwise a year outside the window.
- AT uses the canonical first place in the city's typical-place set with prob
  `p_redundancy`, otherwise a place outside the set.
Sweep `p_redundancy ∈ {0.2, 0.6, 0.9}`. Report all results per level.

Parameter `p_restate = 0.3` caps repeated identical WORKS/MOVED facts. When a
sampled event would make the realized restatement fraction for either relation
exceed `p_restate`, resample another fact.
Forced initial and superseding events are not restatements. Each world logs the
realized WORKS, MOVED, and combined restatement rates, where a restatement is a
fact tuple `(entity, relation, object)` already observed in an earlier episode.

**Entities.** `n_entities = 400` per world, mapped to a random subset of
E tokens.

**Episodes.** Stream of `n_episodes = 2000`. Each episode: 1–3 entities,
4–8 facts, 0–6 filler tokens, a time bucket. Facts are sampled without
replacement: an episode contains at most one fact for each (entity, relation)
pair. `OWNS` objects use J tokens; P tokens are reserved for places following
`AT`. Fact classes:
core relational (WORKS, MOVED), causal-linked (MOVED follows WORKS),
temporal (JOINED_YEAR), social (TOLD/MET + AT), incidental (COLOR, OWNS).

Rendered form:
```
EP T_12 E_7 WORKS O_3 SEP E_7 MOVED C_14 SEP E_7 TOLD E_22 AT P_5 SEP
E_7 COLOR K_2 F_31 F_9 EOS
```

**Conflicts.** 30% of entities later get a superseding event: a new episode
contains both a new WORKS and a new MOVED. The new MOVED follows the new
company's home-city prior with probability `p_redundancy`. Its time bucket is
strictly later than the original event's bucket. Other facts, including any
TOLD or MET interaction, follow the ordinary episode-sampling process.

**Interference.** 10% of entities have a "sibling" sharing all relation
objects except one slot.

## 4. Queries

Query types, rendered as `Q <operator/args> A` with answer token(s) after A:

| Type | Example | Answer |
|---|---|---|
| A 1-hop | `Q E_7 MOVED A` | C_14 |
| B 2-hop | `Q HOP2 E_7 TOLD AT P_5 TIME T_12 WORKS A` | company of the entity E_7 told at P_5 in T_12, from a distinct earlier episode |
| C conflict | `Q E_7 MOVED PREV A` / `CURRENT` | earlier / latest value |
| E unavailable | `Q E_7 STUDIED A` (never stated) | ABSTAIN |
| R reinstatement | `Q CTX E_7 AT P_5 T_12 MOVED A` | target entity's city, cued by its interaction partner, place, and time; the target is omitted |

Type B targets exclude entities with superseding events, and every B query's
TOLD episode is strictly later than its separate WORKS evidence episode. Type C
is restricted to superseded entities and lists every prior episode containing
the queried relation for that entity as evidence.

For type R, the evidence episode contains a TOLD or MET interaction and the
other participant's MOVED fact. The cue is the partner entity, interaction
place, and episode time bucket; the full cue identifies exactly one stream
episode. Each R query records `competitor_count`, the number of earlier episodes
whose interaction cue matches exactly two of those three features, and
`from_superseding`, a boolean indicating whether the target's MOVED fact is in a
same-episode WORKS+MOVED change from earlier values.

**Demand.** Per world, the mixture over (query type × fact class) is drawn
from Dirichlet(α=0.5), then each query type's aggregate share is floored at
8% and the mixture is renormalized. At stream midpoint the Dirichlet mixture
is redrawn and floored again. Query lag is heavy-tailed (log-uniform over
1..stream length); ≥10% of queries target items older than 90% of the stream.

Queries are interleaved: after each episode, emit Poisson(λ=3.0) queries.
Evaluation set: ≥5,000 queries per world. Eval worlds target at least 300 unique
forms for A, B, C, and E; forms are distinguished by their complete token
sequence. B includes the interaction time bucket only. R queries use
unique supported cues and must provide at least 250 unique forms at every
`p_redundancy` level. Calibration samples up to 300 unique forms per type
without replacement; it requires 300 for A/B/C/E and 250 for R.

## 5. Abstention labels

World query labels come only from world ground truth. A query for an
`(entity, relation)` with no fact anywhere in the world is type E with target
ABSTAIN; the grammar never creates a value label for an absent world fact.

Reader-training labels additionally depend on the sampled context. If the
world fact exists and its value or sufficient prior-predictive evidence is in
context, the target is the world's value. If the world fact exists but neither
the value nor predictive evidence is in context, the training target is
ABSTAIN. Thus predictive evidence controls whether a known world value is
supervised, but never invents a value absent from the world.

## 6. Reader

Decoder-only transformer, pre-norm RMSNorm, RoPE, no dropout, tied
embeddings, loss on answer tokens only.

Calibration sizes (approximate non-embedding params):

| name | layers | d_model | heads | ffn | params |
|---|---|---|---|---|---|
| r2m | 4 | 192 | 6 | 768 | ~1.8M |
| r5m | 6 | 256 | 8 | 1024 | ~4.7M |
| r14m | 8 | 384 | 8 | 1536 | ~14M |
| r31m | 10 | 512 | 8 | 2048 | ~31M |

Context length 384. AdamW (β 0.9/0.95, wd 0.1), peak lr 6e-4 (scale ∝
1/sqrt(d_model/256) for larger sizes), 1k warmup, cosine to 10%, batch 64,
grad clip 1.0. Start with 15k steps; extend only if validation loss is still
falling by >1% per 1k steps.

**Training data** is generated on the fly from reader-training worlds:
each sequence = shuffled context items + 1 query + answer. Context items mix
**every representation format any condition can produce** (full episodes,
fact-only, residual-coded, gist traces at every fidelity level, summaries),
so no condition is out of distribution for the reader. Include
distractor items and cases where evidence is missing (→ ABSTAIN).

Entity remapping is redrawn for every training sequence.

## 7. Memory representations (render.py)

Continuous fidelity r ∈ [0,1], quantized to levels for rendering:

| level | content |
|---|---|
| L4 | full episode incl. filler |
| L3 | all facts, no filler |
| L2 | first `ceil(0.6 × n_facts)` facts in canonical episode order; explicit placeholder until utility weights are available |
| L1 | time, subject entities, and grouped relation keys; all values omitted: `GIST T_0 E_420 SIBLING E_23 SEP E_312 WORKS MOVED MET E_420 SEP E_23 TOLD E_312 EOS` |
| L0 | deleted |

Headers are format-specific: L4/L3 use `EP`, partial L2 episodes use `EPP`,
residual-coded episodes use `RES`, L1 uses `GIST`, and summary lines use
`SUM`.

Until calibrated utility weights exist, L2 uses the deterministic placeholder
rule: retain the first `ceil(0.6 × n_facts)` facts in the episode's canonical
fact order. It does not rank or filter by the currently configured weights.

L1 gist traces retain the time bucket, every subject entity (including
interaction partners), and every relation token. All value tokens (`C_`, `O_`,
`P_`, `Y_`, `K_`, `J_`) are dropped without MASK placeholders. Relations are
grouped by subject in first-appearance order; each group begins with its entity
token, and SEP separates groups. Thus all `(entity, relation)` pairs can be
recovered while values are unavailable.

Residual-coded rendering (condition 4) preserves fact presence and replaces a
prior-predicted value with `PRED`, for example `E_7 MOVED PRED` or
`E_208 TOLD E_9 AT PRED`. Incidental COLOR and OWNS facts are never altered.
Decoding `PRED` with the deterministic shared prior and conditioning facts
reproduces the original fact set exactly.

Summary rendering (condition 10): for a stream prefix ending at episode `i`,
emit one `SUM` line per entity containing its latest value for each relation
seen through `i`. Summary snapshots are rewritten every 100 episodes; the
snapshot after the 500th episode therefore includes episode ids 0 through 499.

## 8. Retriever (shared)

Item keys = set of non-filler tokens in the rendered item. Score = count of
query tokens (excluding Q/A/operators) in key, ties broken by recency.
Fill context greedily by score until K tokens. Retrieval ops = candidates
scored × key length (counted, not timed).

## 9. Conditions (conditions.yaml)

1. `none` — empty context.
2. `full_oracle` — all ground-truth-relevant episodes at L4, unbounded K
   (ceiling).
3. `raw_offload` — store everything at L4; retrieve top-k to K.
4. `residual_offload` — store everything residual-coded; retrieve to K.
5. `evict_{fifo,lru,freq}` — store budget B items at L3; retrieve to K.
   Report the best of the family.
6. `evict_learned` — per-item utility regressor (logistic, features:
   age, use count, fact class, uniqueness) trained on policy-tuning worlds;
   store budget B.
7. `progressive` — each item has r_i; r decays by rule (below), renders at
   its level; items with r_i < τ are latent and excluded from retrieval
   unless the cue's candidate-set size ≤ c (cue gating); successful
   retrieval refreshes r_i.
8. `progressive_nogate` — 7 without cue gating.
9. `progressive_uniform` — 7 with identical decay for all fact classes
   (no utility model).
10. `summary` — summaries only.
11. `raw_offload_qrender` — store raw L4 episodes and retrieve with the same
    shared item scorer, but render each selected item with only facts whose
    `(entity, relation)` or interaction-cue features match the query. The
    persistent store and item keys remain raw L4; only the retrieved view is
    query-rendered. This isolates query-conditioned rendering from storage
    compression.

Decay rules for 7 (run as sub-conditions): `time`, `recency_of_use`,
`use_count`, `expected_utility` (fact-class weights from policy-tuning
worlds). Learned decay is out of scope for Stage 0.

Store budget B for 5–6 and the progressive store are matched in persistent
tokens to the progressive condition's store at each sweep point.
Condition 11 is reported alongside `raw_offload` in the oracle-reader
pre-check as the query-rendered raw baseline.

## 10. Accounting record (per query, aggregated per unit)

`context_tokens`, `retrieval_ops`, `reconstruction_passes` (reader forward
passes; normally 1), `store_tokens`, `store_bits` (tokens × log2|V|, plus
metadata at 32 bits per field per item), `maintenance_ops_per_step`,
`policy_params`, `wallclock_ms` (sanity only).

## 11. Scoring

Per query: +1 correct (including ABSTAIN on type E), 0 ABSTAIN on an
answerable query, −1 wrong non-abstain answer (confabulation).
Utility U = mean score. Also report plain accuracy and confabulation rate.

## 12. Budget sweep

K ∈ {384, 256, 128, 64, 32, 16}. Store budgets for eviction conditions
matched per point.

## 13. Statistics

Unit = (world, reader_seed). All comparisons are paired within unit.
Pilot: 4 worlds × 3 seeds. Fit `U ~ condition + (1|world) + (1|seed:world)`;
simulate power for the pre-registered minimum effect; fix N before the main
run. CIs by hierarchical bootstrap (resample worlds, then seeds within
world), 10,000 resamples. Paired permutation test across units for p.

Frontier metric: per unit, interpolate U(K) monotonically per condition;
K* = budget achieving U = 0.9 × U(full_oracle); report mean log-ratio of K*
between conditions with bootstrap CI, at 0.8/0.9/0.95.

## 14. Figures

Fig 1: U vs K (log x), one line per condition, bootstrap bands, faceted by
p_redundancy.
Fig 2: reinstatement — retrieval probability and correctness vs cue
candidate-set size (1/2/4/8), by fidelity level; confabulation dashed.
