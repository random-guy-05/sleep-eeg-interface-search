import unittest
import numpy as np
import pandas as pd
from cinc_trainable_transfer import choose_penalty,label_split,design_split


class TransferGuards(unittest.TestCase):
    def test_penalty_ties_prefer_stronger_regularization(self):
        self.assertEqual(choose_penalty([.6,.6,.6,.6]),3)
        self.assertEqual(choose_penalty([.7,.6,.6,.6]),0)

    def test_reference_labels_do_not_use_held_outcomes(self):
        f=pd.DataFrame({'target':[1.,2.,3.,4.,5.,6.]});train=np.arange(4);test=np.array([4,5])
        a,b,r=label_split(f,train,test,['target']);f.loc[test,'target']=-100
        a2,b2,r2=label_split(f,train,test,['target']);np.testing.assert_array_equal(a,a2);self.assertEqual(r,r2);self.assertFalse(np.array_equal(b,b2))

    def test_target_features_do_not_change_training_design(self):
        n=20;f=pd.DataFrame({'age':np.arange(n)+20.,'sex':np.arange(n)%2,'education':['college']*n,'bmi':[25.]*n,'ahi':[10.]*n,'ess':[6.]*n,'phq9':[2.]*n});x=np.random.default_rng(44).normal(size=(n,4));tr=np.arange(16);te=np.arange(16,20)
        a,b=design_split(f,x,tr,te,True);x[te]*=1000;aa,bb=design_split(f,x,tr,te,True)
        np.testing.assert_array_equal(a,aa);self.assertFalse(np.array_equal(b,bb))


if __name__=='__main__':unittest.main()
