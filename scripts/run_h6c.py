#!/usr/bin/env python3
"""Command-line entry point for H6C constitutive identification."""

import argparse
from dataclasses import replace
from pathlib import Path

from flesh_and_bone.h6c_experiment import H6CConfig, run_h6c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/runs/h6c_final"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()
    config = replace(
        H6CConfig(), device=args.device, cycles=args.cycles
    )
    report = run_h6c(args.output, config=config)
    print("PASS" if report["acceptance"]["pass"] else "FAIL")


if __name__ == "__main__":
    main()
