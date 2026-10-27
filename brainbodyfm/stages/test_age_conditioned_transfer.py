"""Synthetic-only checks; these never read STAGES cognition or scores."""
import unittest
import numpy as np
import pandas as pd
from age_conditioned_transfer import eligible_pairs, rank_auc, continuous_concordance, fixed_score, outcome_z, CENTERS


class RankEvaluationSafeguards(unittest.TestCase):
    def test_official_pair_win_tie_and_inclusive_age_window(self):
        age = np.array([40., 42., 43., 70., 40.])
        center = np.array(['a', 'a', 'a', 'a', 'b'])
        labels = np.array([True, False, False, True, False])
        score = np.array([1., 1., 0., 3., 10.])
        pairs = eligible_pairs(age, center)
        auc, n = rank_auc(labels, score, pairs)
        self.assertEqual(n, 1)
        self.assertEqual(auc[0], .5)

    def test_agrees_with_literal_challenge_definition(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            age = rng.integers(30, 45, 50)
            center = rng.choice(['a', 'b'], 50)
            labels = rng.integers(0, 2, 50).astype(bool)
            score = rng.integers(0, 4, 50)
            wins = count = 0
            for i in np.flatnonzero(labels):
                for j in np.flatnonzero(~labels):
                    if center[i] == center[j] and abs(age[i] - age[j]) <= 2:
                        count += 1
                        wins += (float(score[i] > score[j]) + .5 * (score[i] == score[j]))
            actual, n = rank_auc(labels, score, eligible_pairs(age, center))
            self.assertEqual(n, count)
            self.assertAlmostEqual(actual[0], wins / count)

    def test_empty_pair_set_is_not_chance(self):
        value, n = rank_auc(np.array([1, 1]), [1., 2.], (np.array([0]), np.array([1])))
        self.assertEqual(n, 0)
        self.assertTrue(np.isnan(value[0]))

    def test_frozen_orientation_and_scale_match_entry(self):
        x = np.array([[1., 2., 3., 4.], [5., 4., 2., 0.]])
        norm = dict(feat_mean=[1., 1., -1., -1.], feat_scale=[2., 3., 4., 5.])
        expected = ((x * [1, 1, -1, -1] - norm['feat_mean']) / norm['feat_scale']) @ [.36, .27, .235, .135]
        np.testing.assert_array_equal(fixed_score(x, norm), expected)

    def test_continuous_pairs_require_quarter_sd_separation(self):
        z = np.array([0., .1, 1.])
        pairs = (np.array([0, 0, 1]), np.array([1, 2, 2]))
        self.assertEqual(continuous_concordance(z, np.array([3., 2., 1.]), pairs), 1.)

    def test_outcome_adjustment_is_score_free_and_refitted(self):
        rng = np.random.default_rng(4); n = 100
        f = pd.DataFrame(dict(age=rng.uniform(25, 75, n), sex=rng.integers(0, 2, n),
                              center=np.repeat(CENTERS, 25), education=rng.choice(['unknown', 'college'], n),
                              executive=rng.normal(size=n), memory=rng.normal(size=n)))
        z = outcome_z(f, ['executive', 'memory'])
        f['unrelated_model_score'] = rng.normal(size=n) * 1e9
        np.testing.assert_array_equal(z, outcome_z(f, ['executive', 'memory']))
        self.assertAlmostEqual(z.std(), 1.)
        f.loc[0, 'executive'] += 5
        self.assertFalse(np.allclose(z, outcome_z(f, ['executive', 'memory'])))


if __name__ == '__main__':
    unittest.main()
