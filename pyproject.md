# pyproject.toml

## Purpose

Defines the installable `flesh-and-bone` Python package, its runtime
dependencies, and the CPU-fast pytest discovery path. H4 adds SciPy spatial/
component operations and Trimesh surface voxelization.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment scripts | PyTorch, NumPy, Pillow, SciPy, and Trimesh are installed | Dependency removal |
| Tests | `src` is importable and tests live under `tests/` | Package/test path changes |
| Packaging | Setuptools discovers `src/flesh_and_bone` | Layout changes |
