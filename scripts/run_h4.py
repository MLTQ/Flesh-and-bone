#!/usr/bin/env python3
"""Command-line entry point for H4 production-rig validation."""

import argparse
from dataclasses import replace
from pathlib import Path

from flesh_and_bone.h4_experiment import H4Config, run_h4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/runs/h4_final"),
    )
    parser.add_argument("--archive")
    parser.add_argument("--rig-asset")
    parser.add_argument("--pitch", type=float, default=0.025)
    args = parser.parse_args()
    config = H4Config(pitch=args.pitch)
    if args.archive:
        config = replace(config, archive_path=args.archive)
    if args.rig_asset:
        config = replace(config, rig_asset_path=args.rig_asset)
    report = run_h4(args.output, config)
    print("PASS" if report["acceptance"]["pass"] else "FAIL")
    print(f"selected influences: {report['selected_influence_count']}")
    print(f"volume cells: {report['volume']['cell_count']}")


if __name__ == "__main__":
    main()
