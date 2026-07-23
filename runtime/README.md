# Native runtime assets

`runtime/Assets/` is a generated, ignored staging directory for the native
Flesh & Bone Lab. It contains three physical body profiles and one frozen H7C
mechanics model:

| Profile | Pitch | Cells |
| --- | ---: | ---: |
| coarse | 25.0 mm | 13,273 |
| dense | 17.5 mm | 35,527 |
| ultra | 12.5 mm | 91,979 |

Run `python scripts/export_flesh_runtime.py` to recreate the binary `FNB1` body
files and `FNM1` model file from the curated experiment state. Run
`scripts/make_flesh_app.sh` to export, compile, and place a self-contained
`FleshAndBoneLab.app` under `dist/`.

The body files include a fixed six-neighbor graph, material data, six skinning
influences padded to eight lanes, the tracked walk matrices, and a deterministic
render sample order. They are deliberately not source-of-truth evidence; the
experiment assets and checkpoints from which they are derived remain the
authoritative records.
