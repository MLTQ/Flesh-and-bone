# test_retarget_skin.py

## Purpose

Locks the coordinate-system closure check and periodic indexing used before a
Kimodo motion is allowed into flesh evaluation.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Identity rest | Canonical identity locals yield identity skin and rest endpoints | Bind/basis/order error |
| Palindrome | `0..N-1,N-2..1` with no duplicate turn/wrap endpoint | Acceleration discontinuity from duplicate frames |
