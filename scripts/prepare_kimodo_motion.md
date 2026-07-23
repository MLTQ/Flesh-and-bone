# prepare_kimodo_motion.py

## Purpose

Uses Archepelago's validated SOMA retargeter to convert a raw Kimodo NPZ into a
portable periodic LBS/bone cycle for Flesh and Bone's H6K evaluation.

## Components

### `_destination_rig`
- **Does**: Converts H4 rest bone heads/parents/names into Archepelago's
  canonical identity-rest `Rig` offsets.

### `main`
- **Does**: Profiles the named Meshy rig, retargets raw SOMA rotations, converts
  canonical motion to H4 skin matrices, numerically checks identity-rest skin,
  skins the dense volume in five-frame chunks, palindrome-closes the first 30
  frames, and saves metadata plus LBS positions and bone endpoints.
- **Rationale**: The script imports Archepelago from an explicit path rather
  than adding it as a runtime dependency of the core package.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Archepelago | `skeletor_motion` canonical retarget API and SOMA skeleton dump | API/layout change |
| H6K runner | `flesh-and-bone-h6-kimodo-motion-v1` with 58 phases by default | Artifact schema |
| Research audit | Prompt, seed, hash, mapping, profile, root scale, and rest identity error recorded | Metadata removal |
