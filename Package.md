# `Package.swift`

## Purpose

Declare the dependency-free macOS 13+ executable target for the native
Flesh-and-Bone Lab. Python research/training remains governed by
`pyproject.toml`; the two build systems intentionally coexist.

## Components

### `FleshAndBoneLab`

- **Does:** Builds the AppKit/Metal interactive runner and headless benchmark.
- **Rationale:** A separate executable avoids coupling Bonsai's dense voxel NCA
  formats to this sparse particle/rig runtime.
- **Population QA:** Includes a headless directed overcapacity/wound benchmark
  using the same brush, controller, and GPU kernels as the viewer.
- **Directed editing:** Keeps screen-space brush selection in a separate
  documented module from AppKit input and Metal population mutation.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `scripts/make_flesh_app.sh` | release binary is `.build/release/FleshAndBoneLab` | Target/name changes |
| SwiftPM | companion Markdown files, including population controls, are excluded from compilation | New docs not added to excludes |

The target also contains a headless offscreen render test so GPU rendering can
be validated without Accessibility or screen-capture permission.
