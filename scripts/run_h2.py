#!/usr/bin/env python3
"""Command-line entry point for H2 and its causal controls."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from flesh_and_bone.h2_experiment import H2Config, run_h2


ARMS = {
    "main": {},
    "nearest_bone": {
        "recruitment": "nearest_bone",
        "nearest_bone_uses_target_density": False,
    },
    "pressure_off": {"pressure_enabled": False},
    "first_contact": {"plastic_material_enabled": False},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/runs/h2")
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capture-every", type=int, default=10)
    args = parser.parse_args()
    selected = ARMS if args.arm == "all" else {args.arm: ARMS[args.arm]}
    summaries = {}
    for name, arm in selected.items():
        config = replace(
            H2Config(), seed=args.seed, device=args.device,
            capture_every=args.capture_every, **arm,
        )
        output = Path(args.out) / f"{name}_seed{args.seed}"
        report = run_h2(output, config)
        summaries[name] = {
            "output": str(output.resolve()),
            "pre_wound": report["pre_wound"],
            "damaged": report["damaged"],
            "repaired": report["repaired"],
            "moving": report["moving"],
            "acceptance": report["acceptance"],
            "elapsed_seconds": report["elapsed_seconds"],
        }
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
