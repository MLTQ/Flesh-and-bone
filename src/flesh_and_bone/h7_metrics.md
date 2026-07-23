# `h7_metrics.py`

## Intent

H7 inherits H5's long-horizon position, drift, amplitude, softness, and finite
state measurements, then adds metrics specific to the new nonlinear mechanism.
The acceptance function is a direct encoding of the gates frozen in
`experiments/H7.md`.

## Density measurements

Teacher density acceleration is reported as RMS vector norm and maximum vector
norm. The learned rollout independently reports those quantities from every
autoregressive substep, so respecting the cap is not inferred from architecture
alone.

Compression error samples every eighth directed graph edge and recomputes
five-percent excess compression in the moving LBS frame for every visible phase
and cycle. Sampling is deterministic. The target is the nonlinear teacher's
phase state, not the backbone state.

## Causal interpretation

The density-blind control is the exact frozen H6C backbone. The nonlinear task
must produce at least 0.2 mm backbone position error, or it is declared
vacuous. A pass then requires the hybrid to remove at least 60% of that position
error and at least 50% of compression error, in addition to absolute rollout
and safety gates. Ratios and reductions are emitted alongside booleans so a
borderline or degenerate pass remains visible.
