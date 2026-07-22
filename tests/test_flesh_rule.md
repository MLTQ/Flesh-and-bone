# test_flesh_rule.py

## Purpose

Locks H5's feature width/order and checkpointed normalization behavior.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Feature assembly | Five vectors then bone distance/stiffness produce 17 channels | Training/rollout order mismatch |
| Normalized rule | Raw physical inputs return finite three-vector output | Buffer/broadcast failure |
