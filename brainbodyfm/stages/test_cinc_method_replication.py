"""Checks for exact score replication, held-center isolation, and inference."""
import os
from pathlib import Path
import unittest
import numpy as np
import cinc_method_replication as c


class CinCMethodChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=os.environ.get('CINC_ENTRY_SOURCE')
        if not folder:raise unittest.SkipTest('Set CINC_ENTRY_SOURCE to verified submission snapshot')
        c.load_entry(Path(folder))

    def test_exact_submitted_recipe_and_diagnostics(self):
        rng=np.random.default_rng(31);train=rng.normal(size=(80,4));test=rng.normal(size=(25,4))
        sex=rng.integers(2,size=80);held_sex=rng.integers(2,size=25)
        scores,artifact=c.fixed_fold_scores(train,test,sex,held_sex)
        z=(test-train.mean(0))/train.std(0)
        np.testing.assert_allclose(scores[:,0],z@c.SIGNED,atol=1e-12)
        np.testing.assert_allclose(scores[:,1],z@np.sign(c.SIGNED),atol=1e-12)
        for i in range(4):
            w=c.SIGNED.copy();w[i]=0.
            np.testing.assert_allclose(scores[:,i+2],z@w,atol=1e-12)
            np.testing.assert_allclose(scores[:,i+6],z[:,i]*np.sign(c.SIGNED[i]),atol=1e-12)

    def test_held_center_does_not_set_normalization(self):
        rng=np.random.default_rng(45);train=rng.normal(size=(50,4));held=rng.normal(size=(20,4))
        _,a=c.fixed_fold_scores(train,held,np.zeros(50),np.zeros(20))
        _,b=c.fixed_fold_scores(train,held*1000+77,np.zeros(50),np.zeros(20))
        np.testing.assert_array_equal(a.feat_mean,b.feat_mean)
        np.testing.assert_array_equal(a.feat_scale,b.feat_scale)

    def test_classifier_recovers_strong_synthetic_signal(self):
        rng=np.random.default_rng(50);x=rng.normal(size=(120,2));y=x[:,0]>0
        scores=c.classifier_predict(x,y,rng.uniform(20,80,120),x)
        self.assertGreater(np.corrcoef(scores,x[:,0])[0,1],.95)

    def test_holm_correction(self):
        np.testing.assert_allclose(c.holm([.04,.01,.03]),[.06,.03,.06])

    def test_missing_center_is_not_silently_omitted(self):
        values=np.array([.7,.8,np.nan,.6])
        self.assertTrue(np.isnan(values.mean()))
        ci=c.intervals([.7,.8,np.nan,.6])
        self.assertEqual(ci['valid_replicates'],3)
        self.assertEqual(ci['requested_replicates'],4)


if __name__=='__main__':unittest.main()
