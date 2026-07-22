# test_contracts.py

## Purpose

Provides CPU-fast contracts for H topology, bone frames, rest deformation,
checker identity, H0 unique reservations, persistent differentiation, the inert
neural residual, and basic dynamics convergence/mass accounting. H1 contracts
also prove that continuous recruitment does not populate target-site IDs, that a
contact-derived bone embedding reconstructs the contact point, and that the
deficit field moves freely fed cells toward tissue. Texture contracts cover the
plastic-to-locked phenotype transition and a spatial field metric that rejects
a globally balanced but inverted checker.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H0 implementation | All tests pass without GPU or generated artifacts | Geometry/state/dynamics semantics |
| H1 work | Zero-initialized neural residual remains exactly inert | Initialization/output contract |
| H1 continuous recruitment | `assigned_site` stays `-1`; deficit motion approaches tissue | Accidental reintroduction of hard niches or broken field dynamics |
| H1 bone attachment | Contact-derived soft embedding reconstructs the cell position | Motion would tear committed material away from its scaffold coordinates |
| H1 material maturation | Plastic phenotype follows the local body material field and stops changing after lock | Checker identity drifts or repaints during motion |
| Checker evidence | Correct texture scores 1; inverted texture scores 0 | Histogram-only metrics masking spatial collapse |
