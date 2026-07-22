#!/usr/bin/env python3
"""Command-line entry point for H5 learned local flesh mechanics."""

import argparse
from dataclasses import replace
from pathlib import Path

from flesh_and_bone.flesh_rollout import FleshTrainingConfig
from flesh_and_bone.flesh_teacher import ElasticTeacherConfig
from flesh_and_bone.h5_experiment import H5Config, run_h5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/runs/h5_final"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--volume")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--neighbor-coupling", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--render-splat-radius-scale", type=float)
    parser.add_argument("--render-opacity", type=float)
    args = parser.parse_args()
    config = replace(
        H5Config(), device=args.device, seeds=tuple(args.seeds)
    )
    if args.volume:
        config = replace(config, volume_path=args.volume)
    if args.image_size is not None:
        config = replace(config, image_size=args.image_size)
    if args.render_splat_radius_scale is not None:
        config = replace(
            config,
            render_splat_radius_scale=args.render_splat_radius_scale,
        )
    if args.render_opacity is not None:
        config = replace(config, render_opacity=args.render_opacity)
    training = FleshTrainingConfig()
    if args.steps is not None:
        training = replace(training, steps=args.steps)
    if args.batch_size is not None:
        training = replace(training, batch_size=args.batch_size)
    teacher = ElasticTeacherConfig()
    if args.neighbor_coupling is not None:
        teacher = replace(
            teacher, neighbor_coupling=args.neighbor_coupling
        )
    report = run_h5(
        args.output,
        config,
        teacher_config=teacher,
        training_config=training,
    )
    print("PASS" if report["acceptance"]["pass"] else "FAIL")
    for run in report["runs"]:
        print(
            run["seed"],
            f"nrmse={run['training']['holdout_acceleration_nrmse']:.4f}",
            f"rollout={run['learned_rollout']['position_rms']:.5f}",
        )


if __name__ == "__main__":
    main()
