# Repository history

This repository was developed privately and published after Stage 0
concluded. Before publication, the history was rewritten once (October 2026):

- Every commit was re-signed with the author's Aftertrace identity. Author
  and commit dates were kept exactly as recorded.
- Two internal working notes (`AGENTS.md`, an instruction file for coding
  tools, and `docs/TASKS.md`, a task checklist) and a stray `.DS_Store` were
  removed from every commit. Neither note contained study code, data, the
  specification, the pre-registration or results. A commit that only edited
  `AGENTS.md` disappeared as a result.
- No other file was changed in any commit. The specification,
  pre-registration and its amendments, code, configs and results are
  byte-for-byte what was committed at the time.

Rewriting changes commit IDs. Files written before publication, such as
`RESULTS.md`, the pre-registration's final amendment,
`docs/runs/oracle_frontier_v1_PROVENANCE.md` and the run
`git_provenance.json` files, cite the original IDs. Use this table to find
the published commit.

| Original | Published | Commit |
|---|---|---|
| `514c021` | `af5f0ac` | Spec and pre-registration |
| `ef47efc` | `6f8eb4e` | Day 1 complete |
| `3006f18` | `bea297c` | Stage 0 world frozen |
| `7459236` | `e19af29` | Oracle harness + prereg oracle pre-check |
| `097b931` | (removed) | Edited only an internal working note |
| `072b0f9` | `9946b83` | Oracle frontier v1 (code as run) |
| `4d8c46d` | `b4d2369` | Record oracle frontier v1 provenance |
| `be2c315` | `119cfd7` | Record run Git provenance and guard dirty worktrees |
| `98b79e3` | `7d20c29` | Add charged two-hop retrieval to frontier |
| `edc040f` | `5d18d6c` | Keep symbolic oracle monotone on partial R cues |
| `4920c57` | `03bd4ae` | Balance policy tuning on eval query strata |
| `c9f26d1` | `39c5b6a` | Make summary evidence conservative in oracle reads |
| `6965b65` | `024c85d` | Apply oracle frontier censoring rule literally |
| `ccf6099` | `4f57f88` | Make retrieval ordering deterministic across hash seeds (v2 run) |
| `efc25df` | `4e0c328` | Add matched-K fallback amendment after v2 |
| `b064dc6` | `bcb150c` | Fix two-hop evidence retrieval and budget accounting |
| `d88bc78` | `16e12ab` | Charge qrender retrieval on rendered context |
| `ae0a4fb` | `61a60b8` | Report matched-K bootstrap and qrender token efficiency (v3 run) |
| `6678ba6` | `5a8a6e2` | Conclude Stage 0: STOP |
| `53924d4` | `28f5579` | Record Stage 0 STOP amendment (tag `stage0-stop`) |
| `708afe5` | `2692f93` | Stop tracking .DS_Store |

Full 40-character IDs for the runs: v2 ran at `ccf609911007c21db894f98cfc01d9b26dc497c4`
(published `4f57f881541f85caf7439cdd43c6fa7714bc6d4d`); v3 ran at
`ae0a4fbdecabd1f55b5f221f03c75ddad4feb375` (published
`61a60b84809014677c4670818e9973e436b9440c`).
