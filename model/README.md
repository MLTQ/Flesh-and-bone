# Model assets

`Meshy_AI_Blonde_female_mechani_biped.zip` is immutable H4 source input. Its
SHA-256 is
`1fd02f1e7ab1f6fb5771492145054020ca00cd84729be744f4767894dc59f188`.

`derived/meshy_blonde_h4_rig.npz` is the Blender-independent canonical rig,
mesh, UV, weight, animation-matrix, endpoint, and evaluated-vertex oracle. It
can be regenerated with Blender 5.1+:

```bash
blender --background --factory-startup \
  --python scripts/extract_h4_asset.py -- \
  --archive model/Meshy_AI_Blonde_female_mechani_biped.zip \
  --output model/derived/meshy_blonde_h4_rig.npz
```

`derived/meshy_blonde_h4_volume.npz` is the frozen 0.025 m, top-6,
variable-radius volume target. The complete configuration, measurements, and
visual evidence are recorded in `experiments/H4.md` and the ignored generated
run directory `experiments/runs/h4_final/`.

Derived files are reproducible evidence inputs, not hand-authored model source.
