# `make_flesh_app.sh`

## Purpose

Produce a double-clickable, ad-hoc-signed Flesh-and-Bone Lab app with all three
generated body profiles and the frozen H7C model embedded.

## Components

### build pipeline

- **Does:** Exports runtime assets, builds release Swift, assembles the app
  bundle, copies the tracked `runtime/Info.plist` and generated resources, and
  signs locally.
- **Interacts with:** `export_flesh_runtime.py` and `Package.swift`.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| user | output is `dist/FleshAndBoneLab.app` | Destination/name |
| runtime loader | assets are in `Contents/Resources` | Bundle layout |
| macOS | executable and plist names agree | Bundle metadata |

## Usage

```bash
scripts/make_flesh_app.sh
open dist/FleshAndBoneLab.app
```
