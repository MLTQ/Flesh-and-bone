# `h7_experiment.py`

## Intent

This orchestrator enforces H7's data split as a process boundary. Training-state
generation touches only replay, reverse, and walk-then-hold. Qualification then
trains seed 7 and evaluates only half speed. `run_h7_final` refuses to load the
fast or Kimodo motion unless the persisted qualification report passes.

## Stage artifacts

- `training_dataset.pt` contains equal-sized deterministic samples from each
  training motion; `training_data.json` records their teacher diagnostics.
- `qualification.json` contains seed 7's training fit, half-speed one-step
  diagnostic, backbone/hybrid 20-cycle metrics, and gate verdicts.
- `seed*/density_residual.pt` stores only the bounded residual; the frozen H6C
  coefficients and source report are recorded separately.
- Final per-motion JSON files contain the shared backbone control and all three
  seed reports. `metrics.json` is the aggregate verdict.
- One final-cycle residual tensor per seed/motion is retained for rendering
  without rerunning the expensive rollout.

## Fairness and interpretation

The density-blind control is evaluated once per motion because it is identical
for every seed. Final holdout one-step NRMSE is reported as a diagnostic but is
not a post-hoc gate; the predeclared training NRMSE and rollout gates determine
the verdict. Training examples are balanced by motion rather than raw phase
count. Fast and Kimodo remain inaccessible until qualification passes.

The H6C backbone is reconstructed from the already frozen H6C metrics artifact,
including its coefficient provenance and original held-out NRMSE. H7 does not
refit or tune that backbone.

`h7c_teacher_config` and `h7c_training_config` are the canonical constructors
for the frozen H7C scale. Follow-up audits import these rather than duplicating
large coefficient values in multiple runners.
