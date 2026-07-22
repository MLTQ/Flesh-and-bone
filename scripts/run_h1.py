#!/usr/bin/env python3
"""Command-line entry point for H1 and its causal controls."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from flesh_and_bone.h1_experiment import H1Config, run_h1


ARMS = {
    "main": {"pressure_enabled": True, "recruitment": "deficit"},
    "legacy_texture": {
        "pressure_enabled": True,
        "recruitment": "deficit",
        "plastic_material_enabled": False,
    },
    "pressure_off": {"pressure_enabled": False, "recruitment": "deficit"},
    "nearest_bone": {"pressure_enabled": True, "recruitment": "nearest_bone"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/runs/h1")
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--motion-start", type=int, default=400)
    parser.add_argument("--capture-every", type=int, default=8)
    args = parser.parse_args()
    if not (0 < args.motion_start < args.steps):
        parser.error("motion-start must be between zero and steps")
    selected = ARMS if args.arm == "all" else {args.arm: ARMS[args.arm]}
    summaries = {}
    for name, arm in selected.items():
        config = replace(
            H1Config(), seed=args.seed, device=args.device, steps=args.steps,
            motion_start=args.motion_start, capture_every=args.capture_every,
            **arm,
        )
        output = Path(args.out) / f"{name}_seed{args.seed}"
        report = run_h1(output, config)
        summaries[name] = {
            "output": str(output.resolve()),
            "assembly": report["assembly"],
            "moving": report["moving"],
            "acceptance": report["acceptance"],
            "elapsed_seconds": report["elapsed_seconds"],
        }
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
