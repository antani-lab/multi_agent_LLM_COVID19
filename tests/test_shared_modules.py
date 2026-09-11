import numpy as np

from multi_agent_cxr import metrics, reasoner, statistics


def test_quadratic_weighted_kappa_is_one_for_identical_ratings():
    ratings = [0, 2, 4, 8, 12, 18, 24]
    assert metrics.quadratic_weighted_kappa(ratings, ratings) == 1.0


def test_reasoner_parser_accepts_consistent_structured_output():
    raw = (
        '{"extent_right": 2, "density_right": 2, "extent_left": 1, '
        '"density_left": 1, "mrale_right": 4, "mrale_left": 1, '
        '"mrale_total": 5, "covid_positive": "No", "covid_confidence": 0.7, '
        '"agents_used": ["A2"], "rationale": "Mild bilateral opacity."}'
    )
    fields, parse_error, notes = reasoner.parse_reasoner_output(raw, roster=("A2",))
    assert parse_error is None
    assert notes["complete_fields"] is True
    assert fields["mrale_total"] == 5


def test_holm_bonferroni_is_monotone_in_rank_order():
    adjusted, rejected = statistics.holm_bonferroni([0.01, 0.02, 0.20])
    assert np.all(np.isfinite(adjusted))
    assert 0 <= adjusted[0] <= adjusted[1] <= adjusted[2] <= 1
    assert rejected == [True, True, False]


def test_missing_values_include_nonfinite_numbers():
    assert metrics.is_missing(None)
    assert metrics.is_missing(float("nan"))
    assert metrics.is_missing(float("inf"))
    assert not metrics.is_missing(0)
