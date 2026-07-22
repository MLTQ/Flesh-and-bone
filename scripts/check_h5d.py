#!/usr/bin/env python3
"""Check H5D density scaling against baseline and dense run reports."""

import argparse
import json
from pathlib import Path

from flesh_and_bone.h5d_metrics import evaluate_h5d


def _load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-h4", default="experiments/runs/h4_final/metrics.json"
    )
    parser.add_argument(
        "--dense-h4", default="experiments/runs/h4_dense_0175/metrics.json"
    )
    parser.add_argument(
        "--baseline-h5", default="experiments/runs/h5_final/metrics.json"
    )
    parser.add_argument(
        "--dense-h5", default="experiments/runs/h5d_final/metrics.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/runs/h5d_final/density_acceptance.json"),
    )
    args = parser.parse_args()
    report = evaluate_h5d(
        _load(args.baseline_h4),
        _load(args.dense_h4),
        _load(args.baseline_h5),
        _load(args.dense_h5),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS" if report["gates"]["pass"] else "FAIL")
    print(json.dumps(report["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
