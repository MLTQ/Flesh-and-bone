#!/usr/bin/env python3
"""Command-line entry point for H3 humanoid learned-fate experiments."""

import argparse
from dataclasses import replace
from pathlib import Path

from flesh_and_bone.h3_experiment import H3Config, run_h3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/runs/h3"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--arm",
        choices=("oracle", "learned", "local_deficit", "no_shortage", "all"),
        default="all",
    )
    args = parser.parse_args()

    arms = (
        ("oracle", "learned", "local_deficit", "no_shortage")
        if args.arm == "all" else (args.arm,)
    )
    oracle_report = None
    if any(arm != "oracle" for arm in arms):
        oracle_report = run_h3(
            args.output / f"oracle_seed{args.seed}",
            H3Config(arm="oracle", seed=args.seed, device=args.device),
        )
    for arm in arms:
        if arm == "oracle" and oracle_report is not None:
            continue
        oracle_coverage = (
            oracle_report["pre_wound"]["coverage"]
            if oracle_report is not None else None
        )
        config = replace(
            H3Config(),
            arm=arm,
            seed=args.seed,
            device=args.device,
            oracle_pre_coverage=oracle_coverage,
        )
        report = run_h3(
            args.output / f"{arm}_seed{args.seed}", config
        )
        print(
            arm,
            "PASS" if report["acceptance"]["pass"] else "FAIL",
            f"coverage={report['moving']['coverage']:.4f}",
        )


if __name__ == "__main__":
    main()
