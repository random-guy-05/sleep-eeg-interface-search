"""Leakage and estimability checks for the additional-patient evaluation."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import timing14_fixed_scores as m

class TimingCheck(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(12);centers=m.CENTERS
  self.original=pd.DataFrame({'cohort':['original']*80,'center':np.repeat(centers,20),'age':40+np.tile(np.arange(20)/3,4),'pcet_accuracy2':rng.normal(size=80)})
  self.new=pd.DataFrame({'cohort':['additional']*20,'center':np.repeat(centers,5),'age':40+np.tile(np.arange(5)/3,4),'pcet_accuracy2':rng.normal(size=20)})
  self.oh=rng.normal(size=(80,4));self.nh=rng.normal(size=(20,4));self.ob=rng.normal(size=80);self.nb=rng.normal(size=20)
 def evaluate(self,new=None,nh=None,nb=None):
  return m.evaluate(self.original,self.new if new is None else new,self.oh,self.nh if nh is None else nh,self.ob,self.nb if nb is None else nb,('pcet_accuracy2',))
 def test_reference_never_includes_new_patients_or_held_center(self):
  real=m.source_reference
  def check(source,target,columns):
   self.assertTrue(source.cohort.eq('original').all())
   self.assertEqual(len(set(source.center)&set(target.center)),0)
   return real(source,target,columns)
  with patch.object(m,'source_reference',side_effect=check):self.evaluate()
 def test_duplicate_additional_sample_does_not_change_its_metrics(self):
  a=self.evaluate();b=self.evaluate(pd.concat([self.new,self.new],ignore_index=True),np.tile(self.nh,(2,1)),np.tile(self.nb,2))
  for k in a:
   if k.startswith('additional__') and '__counts' not in k:np.testing.assert_allclose(a[k],b[k],atol=1e-12,equal_nan=True)
 def test_no_pairs_remains_nonestimable(self):
  new=self.new.copy();new['pcet_accuracy2']=100.
  result=self.evaluate(new)
  self.assertEqual(result['additional__counts']['eligible_pairs'],0)
  self.assertTrue(np.isnan(result['additional__brain_health_risk__age_conditioned']))
  self.assertEqual(sum(v['eligible_pairs']for v in result['additional__counts']['by_center'].values()),0)

if __name__=='__main__':unittest.main()
