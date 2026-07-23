# `test_runtime_export.py`

## Purpose

Protect the byte-level contract between Python research artifacts and the
native Swift/Metal runtime without exporting production-size bodies in tests.

## Components

### Graph/influence packing tests

- **Does:** Verifies eight-lane alignment, `-1` neighbor padding, descending
  top-six selection, zero influence padding, exact normalization, and dominant
  bone-segment source projection including endpoint clamping.

### Model binary test

- **Does:** Exports a deterministic synthetic checkpoint and checks magic,
  dimensions, parameter count, and exact payload size.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `runtime_export.py` | v2 body header and fixed array packing | Binary format changes |
| `RuntimeAsset.swift` | learned payload contains 1,314 MLP parameters | Architecture changes |
