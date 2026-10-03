# Oracle frontier v1 provenance

- Run directory: `runs/oracle_frontier_final/`
- Run-time HEAD: `7459236` (`Oracle harness + prereg oracle pre-check`)
- Tree state at run time: dirty; the seven-file diff was uncommitted.
- Reproduction commit: `072b0f9` (`Oracle frontier v1 (code as run): K* censoring; oracle-utility policy tuning; incremental retriever index + key cache; progressive fidelity-step bookkeeping; frontier resume/reporting; summary-snapshot test`). This commit contains the exact code/config changes used for the saved run.
- Decision line, verbatim: `REDUCED: oracle-reader pre-check (censored K* cases=15; included conservatively)`
