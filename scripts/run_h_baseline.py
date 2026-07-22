#!/usr/bin/env python3
"""Command-line entry point for the deterministic H0 control."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from flesh_and_bone.experiment import ExperimentConfig, run_h0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/runs/h0_seed7")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--motion-start", type=int, default=230)
    parser.add_argument("--capture-every", type=int, default=6)
    args = parser.parse_args()
    config = replace(
        ExperimentConfig(), seed=args.seed, device=args.device, steps=args.steps,
        motion_start=args.motion_start, capture_every=args.capture_every,
    )
    if not (0 < config.motion_start < config.steps):
        parser.error("motion-start must be between zero and steps")
    report = run_h0(Path(args.out), config)
    print(json.dumps({
        "output": str(Path(args.out).resolve()),
        "site_count": report["site_count"],
        "assembly": report["assembly"],
        "moving": report["moving"],
        "acceptance": report["acceptance"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
