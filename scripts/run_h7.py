#!/usr/bin/env python3
"""Command-line entry point for staged H7 bounded density experiments."""

import argparse
from dataclasses import replace
from pathlib import Path

from flesh_and_bone.h7_experiment import (
    H7Config,
    h7c_teacher_config,
    h7c_training_config,
    prepare_h7_training,
    run_h7_final,
    run_h7_qualification,
)
from flesh_and_bone.density_rule import DensityTrainingConfig
from flesh_and_bone.density_teacher import DensityTeacherConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "qualification", "final")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/runs/h7_initial")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument(
        "--profile", choices=("initial", "h7b", "h7c"), default="initial"
    )
    args = parser.parse_args()
    config = replace(H7Config(), device=args.device, cycles=args.cycles)
    teacher = DensityTeacherConfig()
    training = DensityTrainingConfig()
    if args.profile == "h7b":
        teacher = replace(
            teacher,
            pressure_near=240.0,
            pressure_far=400.0,
            cohesion_near=48.0,
            cohesion_far=96.0,
        )
        training = replace(
            training,
            pressure_max=480.0,
            cohesion_max=144.0,
        )
    elif args.profile == "h7c":
        teacher = h7c_teacher_config()
        training = h7c_training_config()
    if args.stage == "prepare":
        report = prepare_h7_training(
            args.output, config=config, teacher_config=teacher,
            training_config=training,
        )
    elif args.stage == "qualification":
        report = run_h7_qualification(
            args.output, config=config, teacher_config=teacher,
            training_config=training,
        )
    else:
        report = run_h7_final(
            args.output, config=config, teacher_config=teacher,
            training_config=training,
        )
    print(report.get("acceptance", report.get("stage")))


if __name__ == "__main__":
    main()
