#!/usr/bin/env python3
"""Command-line entry point for H5 learned local flesh mechanics."""

import argparse
from dataclasses import replace
from pathlib import Path

from flesh_and_bone.h5_experiment import H5Config, run_h5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/runs/h5_final"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    args = parser.parse_args()
    config = replace(
        H5Config(), device=args.device, seeds=tuple(args.seeds)
    )
    report = run_h5(args.output, config)
    print("PASS" if report["acceptance"]["pass"] else "FAIL")
    for run in report["runs"]:
        print(
            run["seed"],
            f"nrmse={run['training']['holdout_acceleration_nrmse']:.4f}",
            f"rollout={run['learned_rollout']['position_rms']:.5f}",
        )


if __name__ == "__main__":
    main()
