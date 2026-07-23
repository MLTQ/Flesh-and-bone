"""CPU-fast bridge gate contracts for H6K."""

from flesh_and_bone.h6k_experiment import bridge_acceptance


def test_bridge_acceptance_requires_every_predeclared_conversion_gate():
    metadata = {
        "destination_profile": {
            "body_type": "humanoid",
            "confidence": 1.0,
        },
        "mapped_role_count": 22,
        "root_scale": 0.9,
        "rest_identity_skin_max_error": 1e-7,
        "finite": True,
        "root_xz_travel": 0.08,
    }
    assert bridge_acceptance(metadata)["pass"]
    metadata["rest_identity_skin_max_error"] = 1.1e-5
    result = bridge_acceptance(metadata)
    assert not result["rest_identity_skin"]
    assert not result["pass"]
