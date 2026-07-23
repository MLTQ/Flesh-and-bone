# `export_flesh_runtime.py`

## Purpose

Build the native app's three physical-resolution body assets and shared H7C
model from canonical research artifacts.

## Components

### `main`

- **Does:** Exports 25 mm, 17.5 mm, and 12.5 mm bodies plus checkpoint seed 7,
  then prints count/size diagnostics.
- **Interacts with:** `export_runtime_bundle` in `runtime_export.py`.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `scripts/make_flesh_app.sh` | default output is `runtime/Assets` | Default path |
| native runner | body filenames use cell counts; model is `h7c_seed7.fnm` | Names |
| H8 evidence | checkpoint and metrics paths identify the passing frozen rule | Source defaults |

## Usage

```bash
PYTHONPATH=src python scripts/export_flesh_runtime.py
```

The H7C checkpoint and H8 metrics are generated evidence under ignored run
directories. The exporter fails rather than substituting another model if they
are absent.
