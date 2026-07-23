"""CPU-fast contracts for H7's causal acceptance logic."""

from flesh_and_bone.h7_metrics import acceptance_h7


def test_h7_acceptance_requires_material_causal_improvement():
    teacher = {
        "finite": True,
        "cycle_seam_rms": 0.0,
        "residual_rms": 0.01,
        "density_acceleration_rms": 0.2,
        "density_acceleration_max": 2.0,
    }
    one_step = {"acceleration_nrmse": 0.05}
    backbone = {"position_rms": 0.001, "compression_error_rms": 0.02}
    hybrid = {
        "position_rms": 0.0003,
        "position_p99": 0.001,
        "position_max": 0.003,
        "amplitude_ratio": 1.0,
        "far_near_amplitude_ratio": 2.0,
        "phase_zero_cycle_drift_rms": 0.0001,
        "finite": True,
        "rollout_integrator_finite": True,
        "density_acceleration_max": 2.0,
        "compression_error_rms": 0.005,
    }
    accepted = acceptance_h7(teacher, one_step, hybrid, backbone)
    assert accepted["pass"]
    hybrid["position_rms"] = 0.0008
    rejected = acceptance_h7(teacher, one_step, hybrid, backbone)
    assert not rejected["position_causal"]
    assert not rejected["pass"]
