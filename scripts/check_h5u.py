#!/usr/bin/env python3
"""Check H5U ultra-density and appearance against canonical baselines."""

import argparse
import json
from pathlib import Path

from flesh_and_bone.h5u_metrics import evaluate_h5u


def _load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-h4", default="experiments/runs/h4_final/metrics.json"
    )
    parser.add_argument(
        "--ultra-h4",
        default="experiments/runs/h4_dense_0125_bary/metrics.json",
    )
    parser.add_argument(
        "--baseline-h5", default="experiments/runs/h5_final/metrics.json"
    )
    parser.add_argument(
        "--ultra-h5", default="experiments/runs/h5u_final/metrics.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/runs/h5u_final/ultra_acceptance.json"),
    )
    args = parser.parse_args()
    report = evaluate_h5u(
        _load(args.baseline_h4),
        _load(args.ultra_h4),
        _load(args.baseline_h5),
        _load(args.ultra_h5),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS" if report["gates"]["pass"] else "FAIL")
    print(json.dumps(report["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
