# test_h3.py

## Purpose

Provides CPU-fast contracts for the H3 variable-bone humanoid, explicit
extra-skeletal anatomy, fine/variable splats, dynamic particle attachment
state, and learned fate scorer.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Humanoid scaffold | 15 connected-frame segments and 18 nonempty tissue regions | Hard-coded five-bone assumptions |
| Extra-skeletal anatomy | Nontrivial distant tissue and nonempty cheek wound | Nearest-bone-only envelope |
| Fine splats | Representation gates pass and world radius is below H2 | Cosmetic-only resolution claim |
| Particle material | 15-bone coordinates and regional splat scale are acquired | State/layout mismatch |
| Fate learner | Short training fits shared teacher and hiding shortage changes choice | Dead model or ineffective ablation |

## Notes

The learner test uses a reduced training budget and a 0.90 threshold for suite
speed. The experiment itself predeclares and records at least 0.97 agreement
using the full frozen budget.
