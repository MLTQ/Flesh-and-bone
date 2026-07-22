"""Contracts for H4 influence selection thresholds."""

from flesh_and_bone.h4_metrics import influence_pass, select_influence_count


def _result(count, rms, p99, maximum):
    return {
        "influence_count": count,
        "animation_rms": rms,
        "animation_p99": p99,
        "animation_max": maximum,
        "finite": True,
    }


def test_influence_selection_chooses_smallest_complete_pass():
    results = [
        _result(3, 1e-3, 1e-3, 1e-3),
        _result(4, 1e-4, 3e-3, 1e-3),
        _result(6, 1e-4, 1e-3, 5e-3),
    ]
    assert not influence_pass(results[0])
    assert not influence_pass(results[1])
    assert influence_pass(results[2])
    assert select_influence_count(results) == 6
