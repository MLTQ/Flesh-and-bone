#!/usr/bin/env python3
"""Command-line entry point for the frozen H7D Kimodo excitation audit."""

import argparse
from dataclasses import replace
from pathlib import Path

from flesh_and_bone.h7d_experiment import H7DConfig, run_h7d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/runs/h7d_frozen_stress"),
    )
    parser.add_argument("--checkpoint-directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = replace(H7DConfig(), device=args.device)
    if args.checkpoint_directory:
        config = replace(
            config, checkpoint_directory=args.checkpoint_directory
        )
    report = run_h7d(args.output, config=config)
    print(report["acceptance"])


if __name__ == "__main__":
    main()
