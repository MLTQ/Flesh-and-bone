# test_h5_experiment.py

## Purpose

Locks H5's density-specific render forwarding without running a teacher,
training a rule, or writing media.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Render configuration | Normal and `4x` frames receive configured size, pitch-relative radius, and opacity | H5's sparse hard-coded style silently reused by H5U |
| Rollout configuration | Learned and neighbor-blind arms both receive the physical teacher configuration | Image-size integer accidentally passed as mechanics configuration |
