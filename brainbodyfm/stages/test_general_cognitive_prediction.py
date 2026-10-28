import unittest
import numpy as np
import pandas as pd
from general_cognitive_prediction import source_reference


class SourceOutcomeReferenceChecks(unittest.TestCase):
    def test_held_outcomes_do_not_set_reference_or_source_labels(self):
        rng=np.random.default_rng(63);source=pd.DataFrame(rng.normal(size=(80,2)),columns=['a','b']);target=pd.DataFrame(rng.normal(size=(20,2)),columns=['a','b'])
        za,zb,r=source_reference(source,target,['a','b'])
        other=target*1000+99;za2,zb2,r2=source_reference(source,other,['a','b'])
        self.assertEqual(r,r2);np.testing.assert_array_equal(za,za2)
        self.assertFalse(np.array_equal(zb,zb2))

    def test_reference_does_not_regress_on_demographics(self):
        source=pd.DataFrame({'a':[0.,1.,2.,3.],'age':[20.,30.,40.,50.]});target=pd.DataFrame({'a':[0.,1.],'age':[22.,32.]})
        a,b,r=source_reference(source,target,['a'])
        source['age']*=100;target['age']*=100
        a2,b2,r2=source_reference(source,target,['a'])
        np.testing.assert_array_equal(a,a2);np.testing.assert_array_equal(b,b2);self.assertEqual(r,r2)

    def test_training_composite_is_standardized(self):
        rng=np.random.default_rng(64);source=pd.DataFrame(rng.normal(size=(120,2)),columns=['a','b']);target=source.iloc[:5].copy()
        z,_,_=source_reference(source,target,['a','b'])
        self.assertAlmostEqual(float(z.mean()),0.);self.assertAlmostEqual(float(z.std()),1.)


if __name__=='__main__':unittest.main()
