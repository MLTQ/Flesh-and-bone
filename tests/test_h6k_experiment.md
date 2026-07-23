# test_h6k_experiment.py

## Purpose

Locks the H6K bridge verdict independently from expensive motion and flesh
rollouts.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Conversion gates | One identity-skin failure rejects an otherwise valid bridge | Blended/partial bridge pass |
