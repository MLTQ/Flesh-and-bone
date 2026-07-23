"""CPU-fast contract for H7D's ecological stability verdict."""

from flesh_and_bone.h7d_experiment import ecological_stability_acceptance


def test_ecological_stability_excludes_only_causal_excitation_gates():
    full = {
        "teacher_finite": True,
        "rollout_finite": True,
        "nonvacuous_backbone": False,
        "position_causal": False,
        "compression_causal": False,
        "position_error_reduction": 0.0,
        "compression_error_reduction": 0.0,
        "pass": False,
    }
    assert ecological_stability_acceptance(full)["pass"]
    full["rollout_finite"] = False
    assert not ecological_stability_acceptance(full)["pass"]
