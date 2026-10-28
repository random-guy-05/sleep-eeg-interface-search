import unittest
import numpy as np
import pandas as pd
from unexpected_cognition_discovery import partial_scan,adjust_p,endpoint,ROSTER,select_candidates


class DiscoveryChecks(unittest.TestCase):
    def test_covariate_confounding_removed(self):
        rng=np.random.default_rng(34);z=rng.normal(size=500);x=z+rng.normal(scale=.1,size=500);y=z+rng.normal(scale=.1,size=500)
        naive=partial_scan(x,y,np.ones((500,1)));adjusted=partial_scan(x,y,np.c_[np.ones(500),z])
        self.assertGreater(naive['r'][0],.95);self.assertLess(abs(adjusted['r'][0]),.15)

    def test_sign_and_ties_are_not_flipped(self):
        rng=np.random.default_rng(35);x=rng.normal(size=100);y=-x+rng.normal(scale=.2,size=100)
        self.assertLess(partial_scan(x,y,np.ones((100,1)))['r'][0],-.9)

    def test_adjustments(self):
        np.testing.assert_allclose(adjust_p([.04,.01,.03],'holm'),[.06,.03,.06])
        np.testing.assert_allclose(adjust_p([.04,.01,.03]),[.04,.03,.04])

    def test_selection_does_not_require_positive_results(self):
        f=pd.DataFrame([dict(endpoint='a',endpoint_order=0,marker='stable4',family='fixed_score',p=.7,q=.9,r=-.1),dict(endpoint='b',endpoint_order=1,marker='stable4',family='fixed_score',p=.8,q=.9,r=.05)])
        result=select_candidates(f,'fixed');self.assertEqual(len(result),2);self.assertEqual(result[0]['discovery_direction'],-1)

    def test_rt_and_status_rules(self):
        f=pd.DataFrame({'PCET_A.status':['C1','C1','C1'],'PCET_A.valid_code':['V','F','V'],'PCET_A.skipped':[0,0,0],'PCET_A.PCET_RTCR':[1200,1000,0]})
        entry=next(r for r in ROSTER if r[0]=='executive_slowness');y,keep=endpoint(f,entry)
        np.testing.assert_array_equal(keep,[True,False,False]);self.assertAlmostEqual(y[0],np.log(1200))


if __name__=='__main__':unittest.main()
