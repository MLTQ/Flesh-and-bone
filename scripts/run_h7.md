# `run_h7.py`

Run H7 in three explicit stages:

```bash
python scripts/run_h7.py prepare --device cuda
python scripts/run_h7.py qualification --device cuda
python scripts/run_h7.py final --device cuda
```

The final command raises an error unless `qualification.json` records a pass.
`--cycles` exists for CPU smoke tests; evidence runs must retain the frozen
20-cycle default. The default output is `experiments/runs/h7_initial` so a
failed initial qualification is preserved rather than overwritten by tuning.

`--profile h7b` selects the separately predeclared scale-only successor in
`experiments/H7B.md`. It must use a different output directory, conventionally
`experiments/runs/h7b_initial`, so H7's failed evidence remains immutable.

`--profile h7c` selects the second scale-only successor documented in
`experiments/H7C.md`; use `experiments/runs/h7c_initial` for its independent
artifacts.
