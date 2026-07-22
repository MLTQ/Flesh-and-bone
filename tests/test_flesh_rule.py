"""CPU-fast contracts for H5 feature and normalization state."""

import torch

from flesh_and_bone.flesh_rule import FleshResidualRule, flesh_features


def test_flesh_features_have_declared_order_and_width():
    vector = torch.zeros(2, 3)
    scalar = torch.zeros(2, 1)
    features = flesh_features(
        vector + 1,
        vector + 2,
        vector + 3,
        vector + 4,
        vector + 5,
        scalar + 6,
        scalar + 7,
    )
    assert features.shape == (2, 17)
    assert features[0].tolist() == (
        [1.0] * 3 + [2.0] * 3 + [3.0] * 3
        + [4.0] * 3 + [5.0] * 3 + [6.0, 7.0]
    )


def test_rule_normalization_round_trips_physical_output_shape():
    rule = FleshResidualRule(hidden_channels=8)
    features = torch.randn(32, 17)
    targets = torch.randn(32, 3) * 4 + 2
    rule.set_normalization(features, targets)
    output = rule(features)
    assert output.shape == targets.shape
    assert torch.isfinite(output).all()
