# `run_h7d.py`

Run the predeclared frozen-checkpoint Kimodo excitation audit with:

```bash
python scripts/run_h7d.py --device cuda
```

The default source is `experiments/runs/h7c_initial`; the default independent
output is `experiments/runs/h7d_frozen_stress`. There are intentionally no
options for stress ratio, phase count, cycles, teacher, or model capacity in the
evidence CLI because those values were frozen before the run.
