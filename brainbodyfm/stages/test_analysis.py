"""Tests for numerical equivalence and held-center isolation."""
import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from analyze import fit_outer,ridge_path

def config():
    base = Path(__file__).resolve().parent
    result = json.loads((base / "protocol_template.json").read_text())
    result.update(json.loads((base / "amendment_01.json").read_text())["overrides"])
    return result


class ScientificSafeguards(unittest.TestCase):
    def test_ridge_path_matches_independent_sklearn_fit(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(60,9));other=rng.normal(size=(19,9));y=rng.normal(size=60)
        alphas=[1e-4,0.3,100.]
        result=ridge_path(x,y,other,alphas)
        expected=np.column_stack([Ridge(alpha=a,solver='svd').fit(x,y).predict(other) for a in alphas])
        np.testing.assert_allclose(result,expected,rtol=1e-10,atol=1e-10)

    def test_held_center_labels_cannot_change_tuning_or_predictions(self):
        rng=np.random.default_rng(71);n=80
        frame=pd.DataFrame({'center':np.repeat(['A','B','C','D'],20), 'age':rng.uniform(25,75,n),
                            'sex':rng.integers(0,2,n),'education':rng.choice(['school','college'],n),
                            'bmi':rng.uniform(20,40,n),'ahi':rng.uniform(0,30,n),
                            'ess':rng.integers(0,24,n),'phq9':rng.integers(0,27,n),
                            'pcet_accuracy2':rng.uniform(0,4,n)})
        frame.loc[[2,25],'ahi']=np.nan
        representation=rng.normal(size=(n,4));cfg=config();cfg['ridge_alpha_grid']=[0.01,1,100]
        original=fit_outer(frame,representation,'D',cfg)
        changed=frame.copy();changed.loc[changed.center=='D','pcet_accuracy2']=rng.normal(100,20,20)
        repeat=fit_outer(changed,representation,'D',cfg)
        self.assertEqual(original['alpha'],repeat['alpha'])
        self.assertEqual(original['source_mae'],repeat['source_mae'])
        np.testing.assert_array_equal(original['predictions'],repeat['predictions'])
        self.assertNotEqual(original['target_mae'],repeat['target_mae'])

    def test_target_covariate_extremes_cannot_change_source_selection(self):
        rng=np.random.default_rng(90);n=80
        frame=pd.DataFrame({'center':np.repeat(['A','B','C','D'],20),'age':rng.uniform(20,70,n),
                            'sex':rng.integers(0,2,n),'education':['college']*n,'bmi':rng.uniform(20,40,n),
                            'ahi':rng.uniform(0,30,n),'ess':rng.uniform(0,24,n),'phq9':rng.uniform(0,27,n),
                            'pcet_accuracy2':rng.uniform(0,4,n)})
        rep=rng.normal(size=(n,4));cfg=config();cfg['ridge_alpha_grid']=[.1,10]
        original=fit_outer(frame,rep,'D',cfg)
        changed=frame.copy();changed.loc[changed.center=='D','age']=1e5
        rep2=rep.copy();rep2[frame.center=='D']=1e6
        repeat=fit_outer(changed,rep2,'D',cfg)
        self.assertEqual(original['alpha'],repeat['alpha'])
        self.assertEqual(original['source_mae'],repeat['source_mae'])


if __name__=='__main__':unittest.main()
