import json
import numpy as np
import pytest
from agentic_search.ieee_access_revision.scripts.feedback_ablation import prompts,select,valid_batch
from agentic_search.ieee_access_revision.scripts.statistical_analysis import pair_surface,participant_draws

def test_no_feedback_context_contains_ids_without_metrics():
    rows=[dict(head_ids=['H1','H2','H3','H4'],auc_source1_to_source2=.123456789,
        auc_source2_to_source1=.987654321,mean_source_auc=.555555555,
        min_source_auc=.123456789,direction_gap=.864197532,selected_C=316.227766)]
    _,u=prompts('no_feedback',rows,[], 'system')
    for k in rows[0]:
        if k!='head_ids':assert k not in u and str(rows[0][k]) not in u
    _,u=prompts('feedback',rows,[],'system')
    assert 'auc_source1_to_source2' in u and '.123456789' in u

def test_robust_selector_uses_minimum_and_lexicographic_tie_only():
    rows=[dict(quartet_key='B',min_source_auc=.6,target_auc=1.),dict(quartet_key='A',min_source_auc=.6,target_auc=0.)]
    assert select(rows)['quartet_key']=='A'

def test_batch_rejects_history_and_forbidden():
    batch=[['A','B','C','D'],['A','B','C','E'],['A','B','D','E']]
    good,errors=valid_batch(batch,set('ABCDE'),{'A|B|C|D'},{'A|B|C|E'})
    assert good==[['A','B','D','E']] and len(errors)==3

def test_participant_bootstrap_rebuilds_pairs_and_preserves_identical_arms():
    y=[1,1,0,0];ages=[20,80,20,80];scores=[1,0,1,-1]
    wins,e=pair_surface(y,ages,scores,2.)
    assert e.sum()==2 and wins.sum()==1.5
    draw,invalid=participant_draws({'a':wins,'b':wins},e,np.random.default_rng(12),1000)
    np.testing.assert_array_equal(draw['a'],draw['b'])
    assert invalid>0
    assert set(np.unique(draw['a']))<={.5,.75,1.}

def test_no_pairs_is_an_error():
    with pytest.raises(ValueError,match='No eligible'):
        participant_draws({'a':np.zeros((1,1))},np.zeros((1,1)),np.random.default_rng(0),10)
