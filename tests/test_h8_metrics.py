from flesh_and_bone.h8_metrics import acceptance_h8, aggregate_h8


def _teacher(density=0.08):
    return {
        "finite": True,
        "residual_rms": 0.008,
        "far_near_amplitude_ratio": 1.4,
        "density_acceleration_rms": density,
        "density_acceleration_max": 2.0,
    }


def _rollout(position, compression, final_velocity=0.003):
    return {
        "position_rms": position,
        "position_p99": 0.003,
        "position_max": 0.008,
        "amplitude_ratio": 1.0,
        "far_near_amplitude_ratio": 1.4,
        "final_residual_rms": 0.006,
        "hold_entry_velocity_rms": 0.02,
        "final_velocity_rms": final_velocity,
        "density_acceleration_max": 2.0,
        "compression_error_rms": compression,
        "finite": True,
    }


def test_eligible_variant_requires_position_and_compression_causality():
    one_step = {"acceleration_nrmse": 0.03}
    passing = acceptance_h8(
        _teacher(), one_step, _rollout(0.0002, 0.002), _rollout(0.001, 0.01)
    )
    failing = acceptance_h8(
        _teacher(), one_step, _rollout(0.0007, 0.007), _rollout(0.001, 0.01)
    )

    assert passing["causal_eligible"]
    assert passing["pass"]
    assert not failing["position_causal"]
    assert not failing["compression_causal"]
    assert not failing["pass"]


def test_low_excitation_skips_causal_claim_but_not_safety():
    result = acceptance_h8(
        _teacher(density=0.001),
        {"acceleration_nrmse": 0.03},
        _rollout(0.001, 0.01),
        _rollout(0.0001, 0.001),
    )

    assert not result["causal_eligible"]
    assert result["safety_pass"]
    assert result["pass"]


def test_softness_gate_preserves_teacher_profile_instead_of_imposing_one():
    teacher = _teacher(density=0.001)
    teacher["far_near_amplitude_ratio"] = 1.005
    matching = _rollout(0.001, 0.01)
    matching["far_near_amplitude_ratio"] = 1.006
    distorted = _rollout(0.001, 0.01)
    distorted["far_near_amplitude_ratio"] = 1.3

    accepted = acceptance_h8(
        teacher,
        {"acceleration_nrmse": 0.03},
        matching,
        _rollout(0.0001, 0.001),
    )
    rejected = acceptance_h8(
        teacher,
        {"acceleration_nrmse": 0.03},
        distorted,
        _rollout(0.0001, 0.001),
    )

    assert accepted["rollout_softness"]
    assert accepted["pass"]
    assert not rejected["rollout_softness"]
    assert not rejected["pass"]


def test_aggregate_enforces_final_causal_eligibility_floor():
    eligible = {"safety_pass": True, "causal_pass": True, "causal_eligible": True}
    ecological = {"safety_pass": True, "causal_pass": True, "causal_eligible": False}

    assert aggregate_h8([eligible] * 3 + [ecological] * 3, 3)["pass"]
    assert not aggregate_h8([eligible] * 2 + [ecological] * 4, 3)["pass"]
