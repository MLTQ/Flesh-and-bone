"""Particle-based neural cellular automata around articulated skeletons."""

from .h3_morphology import H3BodyPlan, build_h3_body_plan
from .humanoid_skeleton import HumanoidScaffold
from .morphology import BodyPlan, build_h_body_plan, deform_body_plan
from .particles import ParticleSystem
from .skeleton import HScaffold, ScaffoldFrame

__all__ = [
    "BodyPlan",
    "H3BodyPlan",
    "HScaffold",
    "HumanoidScaffold",
    "ParticleSystem",
    "ScaffoldFrame",
    "build_h_body_plan",
    "build_h3_body_plan",
    "deform_body_plan",
]
