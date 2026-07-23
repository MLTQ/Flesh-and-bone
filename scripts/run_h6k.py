#!/usr/bin/env python3
"""Command-line entry point for Kimodo ecological motion evaluation."""

import argparse
from dataclasses import replace
from pathlib import Path

from flesh_and_bone.h6k_experiment import H6KConfig, run_h6k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/runs/h6k_final"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--motion")
    args = parser.parse_args()
    config = replace(H6KConfig(), device=args.device)
    if args.motion:
        config = replace(config, motion_path=args.motion)
    report = run_h6k(args.output, config=config)
    print(report["verdicts"])


if __name__ == "__main__":
    main()
