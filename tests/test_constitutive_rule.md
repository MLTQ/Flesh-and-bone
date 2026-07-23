# test_constitutive_rule.py

## Purpose

Locks H6C's physical feature basis, axis-equivariant scalar application, and
phase-held-out closed-form coefficient recovery.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Five-term basis | Shape is cell×term×xyz with declared signs/scales | Feature slicing error |
| Shared scalar rule | Coefficients act identically on xyz | Accidental axis-specific fit |
| Identifier | Known coefficients recover through the modulo-five holdout | Normal-equation/split error |
