# pyproject.toml

## Purpose

Defines the installable `flesh-and-bone` Python package, its runtime
dependencies, and the CPU-fast pytest discovery path. H4 adds SciPy spatial/
component operations, Trimesh surface voxelization, and Rtree-backed closest
triangle queries for continuous UV transfer.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment scripts | PyTorch, NumPy, Pillow, SciPy, Trimesh, and Rtree are installed | Dependency removal |
| Tests | `src` is importable and tests live under `tests/` | Package/test path changes |
| Packaging | Setuptools discovers `src/flesh_and_bone` | Layout changes |
