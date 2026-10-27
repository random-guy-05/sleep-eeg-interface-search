"""Safeguards for the post hoc fixed-score implementation."""
import unittest,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from stable4_extension import evaluate,reps_for_split,CENTERS,STABLE4

class FixedInterfaceSafeguards(unittest.TestCase):
    def test_target_features_cannot_change_source_score_or_scale(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(40,4));b=rng.normal(size=40);q=rng.normal(size=(40,4));tr=np.arange(30);te=np.arange(30,40)
        a=reps_for_split(x,b,q,tr,te)
        changed=x.copy();changed[te]=1e8
        z=reps_for_split(changed,b,q,tr,te)
        np.testing.assert_array_equal(a['stable4_score'][0],z['stable4_score'][0])
        self.assertFalse(np.allclose(a['stable4_score'][1],z['stable4_score'][1]))

    def test_held_labels_cannot_change_fixed_score_predictions(self):
        rng=np.random.default_rng(72);n=80
        f=pd.DataFrame({'center':np.repeat(CENTERS,20),'age':rng.uniform(20,75,n),'sex':rng.integers(0,2,n),'education':rng.choice(['school','college'],n),'bmi':rng.uniform(20,40,n),'ahi':rng.uniform(0,30,n),'ess':rng.uniform(0,24,n),'phq9':rng.uniform(0,27,n),'target':rng.uniform(0,4,n)})
        f.loc[[2,25],'ahi']=np.nan
        x=rng.normal(size=(n,8));b=rng.normal(size=n);names=STABLE4+[f'other_{i}' for i in range(4)];selection={c:names[4:] for c in CENTERS}
        base=Path(__file__).parent;protocol=base/'protocol_template.json'
        if not protocol.exists():protocol=base/'protocol_v1.json'
        cfg=json.loads(protocol.read_text());cfg.update(json.loads((base/'amendment_01.json').read_text())['overrides'])
        with patch('stable4_extension.config',return_value=cfg):original=evaluate(f,x,b,names,selection)
        changed=f.copy();changed.loc[f.center.eq(CENTERS[-1]),'target']=rng.normal(100,10,20)
        with patch('stable4_extension.config',return_value=cfg):repeat=evaluate(changed,x,b,names,selection)
        ix=f.center.eq(CENTERS[-1]).to_numpy()
        self.assertEqual(original[4][CENTERS[-1]],repeat[4][CENTERS[-1]])
        np.testing.assert_array_equal(original[3]['stable4_score'][ix],repeat[3]['stable4_score'][ix])
        self.assertNotEqual(original[2]['stable4_score'][CENTERS[-1]],repeat[2]['stable4_score'][CENTERS[-1]])

if __name__=='__main__':unittest.main()
