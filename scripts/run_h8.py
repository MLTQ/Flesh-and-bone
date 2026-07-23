#!/usr/bin/env python3
"""Run H8 qualification or the sealed final streaming-motion suite."""

import argparse
from pathlib import Path

from flesh_and_bone.h8_experiment import H8Config, run_h8_stage


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("qualification", "final"))
    parser.add_argument(
        "--motion",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="exact H8 motion name and retargeted_motion.npz path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/runs/h8_streaming"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _motion_paths(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--motion must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"invalid or duplicate motion name {name!r}")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def main():
    args = _arguments()
    report = run_h8_stage(
        args.stage,
        _motion_paths(args.motion),
        args.output,
        H8Config(device=args.device),
    )
    print(
        f"H8 {args.stage}: "
        f"{'PASS' if report['acceptance']['pass'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
