#!/usr/bin/env python3
"""Command-line entry point for frozen-rule H6M generalization trials."""

import argparse
from dataclasses import replace
from pathlib import Path

from flesh_and_bone.flesh_teacher import ElasticTeacherConfig
from flesh_and_bone.h6m_experiment import H6MConfig, run_h6m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/runs/h6m_final"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--checkpoint-directory")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--neighbor-coupling", type=float, default=1200.0)
    args = parser.parse_args()
    config = replace(
        H6MConfig(),
        device=args.device,
        cycles=args.cycles,
        render_motions=() if args.skip_render else H6MConfig().render_motions,
    )
    if args.checkpoint_directory:
        config = replace(
            config, checkpoint_directory=args.checkpoint_directory
        )
    teacher = ElasticTeacherConfig(
        neighbor_coupling=args.neighbor_coupling
    )
    report = run_h6m(args.output, config, teacher_config=teacher)
    print("PASS" if report["acceptance"]["pass"] else "FAIL")


if __name__ == "__main__":
    main()
