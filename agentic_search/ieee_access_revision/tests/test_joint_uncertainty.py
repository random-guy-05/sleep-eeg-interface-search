import numpy as np
import pytest

from agentic_search.ieee_access_revision.scripts.joint_uncertainty import bootstrap_site


def test_identical_paired_runs_have_exactly_zero_difference():
    # Unequal run-level performance must remain paired under run resampling.
    tensor = np.array([[[0., .5], [1., 1.]], [[1., 1.], [0., .5]]])
    joint, participant, _ = bootstrap_site({"a": tensor, "b": tensor}, np.ones((2, 2)),
        np.random.default_rng(5), draws=1000, groups={"a": "paired", "b": "paired"})
    np.testing.assert_array_equal(joint["a"], joint["b"])
    np.testing.assert_array_equal(participant["a"], participant["b"])


def test_run_variance_is_added_even_when_participant_scores_are_constant():
    tensor = np.array([np.zeros((2, 2)), np.ones((2, 2))])
    joint, participant, _ = bootstrap_site({"a": tensor}, np.ones((2, 2)),
        np.random.default_rng(5), draws=1000)
    np.testing.assert_array_equal(participant["a"], np.full(1000, .5))
    assert np.var(joint["a"]) > .1
    assert set(joint["a"]) == {0., .5, 1.}


def test_deterministic_method_joint_equals_participant_bootstrap():
    tensor = np.array([[[0., .5], [1., 1.]]])
    joint, participant, _ = bootstrap_site({"a": tensor}, np.ones((2, 2)),
        np.random.default_rng(5), draws=1000)
    np.testing.assert_array_equal(joint["a"], participant["a"])


def test_shared_sparse_age_pairs_are_recomputed():
    eligible = np.eye(2)
    tensor = np.array([[[.5, 0.], [0., 1.]]])
    joint, _, rejected = bootstrap_site({"a": tensor}, eligible,
        np.random.default_rng(5), draws=1000)
    assert rejected > 0
    assert set(joint["a"]) <= {.5, .75, 1.}
    with pytest.raises(ValueError, match="No eligible"):
        bootstrap_site({"a": tensor}, np.zeros((2, 2)), np.random.default_rng(5), draws=10)


def test_external_run_weights_preserve_shared_draws_between_hospitals():
    tensor = np.array([np.zeros((2, 2)), np.ones((2, 2))])
    weights = np.tile([[1., 0.], [0., 1.]], (500, 1))
    first, _, _ = bootstrap_site({"a": tensor}, np.ones((2, 2)), np.random.default_rng(3),
        draws=1000, groups={"a": "common"}, shared_weights={"common": weights})
    second, _, _ = bootstrap_site({"a": tensor}, np.ones((2, 2)), np.random.default_rng(99),
        draws=1000, groups={"a": "common"}, shared_weights={"common": weights})
    np.testing.assert_array_equal(first["a"], weights[:, 1])
    np.testing.assert_array_equal(first["a"], second["a"])
