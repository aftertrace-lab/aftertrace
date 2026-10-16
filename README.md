# Aftertrace · Stage 0

**Storage-time forgetting does not beat query-time selection.**
A pre-registered oracle-reader test of progressive memory demotion under a
context-token budget. **Decision: STOP.**

- Plain-language essay: <https://aftertrace.org/essays/why-forgetting-lost>
- Study page: <https://aftertrace.org/research/storage-time-forgetting>
- Full results and decision record: [`RESULTS.md`](RESULTS.md)
- Paper: coming soon

## The question

When an AI system stores memories, should it let old ones fade to their gist,
the way human memory seems to? Or should it keep everything and decide what
matters only once a question arrives?

Concretely: under a fixed per-query context budget (K tokens placed in the
reader's context), does **progressive demotion** — compressing aging episodes
into compact traces, with cue-gated reinstatement — achieve higher utility
than **raw retention with query-time selection**?

## Design in brief

- **Synthetic worlds with known answers.** 400 people, 2,000 episodes per
  world, jobs and moves that change partway through, and a tunable semantic
  prior (`p_redundancy` = 0.2 / 0.6 / 0.9).
- **Five query types:** simple facts (A), two-hop facts (B), superseded
  "changed" facts (C), never-stated facts that require abstention (E), and
  context-cued reinstatement (R). 300 of each are scored per evaluation world.
- **One shared retriever** for every condition; conditions differ only in
  what is stored and how it is rendered.
- **A symbolic oracle reader.** It answers optimally from whatever is in its
  context, so its frontier bounds what any trained reader could do with the
  same memory condition. If demotion loses here, no reader training can
  rescue it.
- **Utility scoring:** +1 correct, 0 abstain, −1 wrong.
- **Six budgets** (K = 16–384), 12 evaluation worlds (4 per redundancy
  level), demotion policies tuned per K on separate policy-tuning worlds.

The full design is in [`docs/SPEC.md`](docs/SPEC.md).

## Result

Progressive demotion loses to the best keep-everything condition at every
budget and redundancy level. Paired-world bootstrap 95% intervals are below
zero everywhere.

| K | progressive | best raw (query-time render) | difference (range over p) |
|---:|---:|---:|---:|
| 384 | 0.67–0.68 | 1.00 | −0.32 to −0.33 |
| 64 | 0.60–0.61 | 1.00 | −0.39 to −0.40 |
| 32 | 0.34–0.35 | 0.99 | −0.64 to −0.66 |
| 16 | 0.20 | 0.77–0.83 | −0.57 to −0.63 |

The mechanism shows up on conflict questions (type C) at K = 64: progressive
0.12, raw whole-episode offload 0.50, raw with query-time rendering 1.00.
Demotion discards the superseded history those questions need. Cue-gated
reinstatement (type R) gave no advantage. Query-time rendering used about
**8.9× fewer tokens per item** than full episodes.

Important caveat: query-time rendering here is symbolic and charged at zero
model cost. A real system would pay for that selection step.

## The decision trail

The rule was committed before any frontier results existed. Each run records
its git commit and refuses to start from a dirty tree unless that is logged.

| Commit | What happened |
|---|---|
| `af5f0ac` | Initial commit: spec and pre-registration |
| `e19af29` | Oracle harness and the oracle pre-check rule (STOP if every K* ratio is < 1.2×) |
| `9946b83`, `b4d2369` | v1 run (code as run) and its provenance → **REDUCED** |
| `4f57f88` | v2 run, after fixing oracle handling of partial cues and deterministic ties → **REDUCED** |
| `4e0c328` | Matched-K fallback amendment, added after v2 and before v3. **Not triggered.** |
| `61a60b8` | v3 run, after fixing two-hop retrieval and query-time render accounting → **STOP** |
| `5a8a6e2`, `28f5579` | Stage 0 concluded and recorded in the pre-registration |

The tag `stage0-stop` marks the concluded state. The formal label moved from
REDUCED to STOP as instrument bugs were fixed; every fix widened the gap
against demotion rather than narrowing it. Amendments are appended to
[`prereg/PREREGISTRATION.md`](prereg/PREREGISTRATION.md) with dates and
reasons; the original text is never edited.

Commit timestamps are recorded by the author's own git; there is no external
timestamping service behind them. Before publication the history was
re-signed and two internal working notes were removed; dates and all study
files are unchanged. Files written before then cite the original commit IDs.
[`HISTORY.md`](HISTORY.md) maps every original ID to its published one.

## Reproduce

No GPU is needed; the Stage 0 reader is symbolic.

```sh
git clone https://github.com/aftertrace-lab/aftertrace.git
cd aftertrace
python3 -m venv .venv          # Python 3.11 or newer
source .venv/bin/activate
python -m pip install -e .
pytest -q

# exact code used for the v3 run
git checkout 61a60b8
python scripts/oracle_frontier.py --output-dir runs/oracle_frontier_v3_rerun
```

The script prints the decision line and writes `frontier.csv`,
`per_type.csv`, `kstar.csv`, `kstar_ratios_by_pair.csv`,
`matched_k_bootstrap.csv` and a figure to the output directory. Configs live
in [`configs/`](configs/). To redraw the published figures from the saved
v3 tables, run `python scripts/plot_stage0_results.py`.

## What's in the repo

| Path | Contents |
|---|---|
| `aftertrace/world/` | World generator, vocabulary, queries, seed ranges |
| `aftertrace/memory/` | Memory conditions, rendering, shared retriever |
| `aftertrace/reader/` | Symbolic oracle reader; trained-reader code for the phase that was not run |
| `aftertrace/eval/` | Evaluation harness |
| `scripts/` | `oracle_frontier.py` (Stage 0), plotting, world inspection; calibration and timing scripts for the unrun reader phase |
| `configs/` | World, conditions and sweep configs |
| `prereg/` | Pre-registration and dated amendments |
| `runs/oracle_frontier_v3/` | Frozen v3 tables behind the published result |
| `runs/oracle_frontier_v2/`, `runs/oracle_frontier_final/` | Earlier v2 and v1 runs, kept for the instrument history |
| `figures/` | Published figures |
| `docs/` | Spec and run provenance |
| `HISTORY.md` | How the published history relates to the original commit IDs |

`matched_k_bootstrap.csv` has one row per (p_redundancy, K) with progressive
utility, the best raw condition, their difference and its 95% interval.

## Limitations

- One synthetic world family; nothing here establishes behavior on natural
  conversations.
- Heuristic policies and an oracle reader only; no trained-reader result.
- Query-time rendering is symbolic and charged at zero model cost.
- Four evaluation worlds per redundancy level.
- Nothing about regimes where the store itself is the expensive resource,
  such as a model's KV cache.

## License and citation

Code is released under the [MIT License](LICENSE). Documentation, figures and
result tables are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Citation metadata is in [`CITATION.cff`](CITATION.cff).

Shikhar Kumar · [Aftertrace](https://aftertrace.org) · hello@aftertrace.org
