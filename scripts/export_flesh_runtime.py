#!/usr/bin/env python3
"""Export the three native runner body profiles and frozen H7C model."""

import argparse
from pathlib import Path

from flesh_and_bone.runtime_export import export_runtime_bundle


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/Assets"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/runs/h7c_initial/seed7/density_residual.pt"
        ),
    )
    return parser.parse_args()


def main():
    args = _arguments()
    manifest = export_runtime_bundle(
        output_directory=args.output,
        checkpoint_path=args.checkpoint,
        metrics_path="experiments/runs/h8_streaming/metrics.json",
        rig_path="model/derived/meshy_blonde_h4_rig.npz",
        volume_paths=(
            "model/derived/meshy_blonde_h4_volume.npz",
            "model/derived/meshy_blonde_h4_volume_p0175.npz",
            "model/derived/meshy_blonde_h4_volume_p0125.npz",
        ),
        texture_archive="model/Meshy_AI_Blonde_female_mechani_biped.zip",
    )
    for body in manifest["bodies"]:
        print(
            f"{body['cell_count']:>6} cells "
            f"{body['bytes'] / 1_000_000:>5.1f} MB "
            f"pitch {body['pitch'] * 1000:.1f} mm"
        )
    model = manifest["model"]
    print(
        f"model {model['learned_parameters']} learned parameters "
        f"{model['bytes'] / 1000:.1f} KB"
    )


if __name__ == "__main__":
    main()
