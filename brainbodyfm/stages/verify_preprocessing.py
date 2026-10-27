"""Compare the bounded implementation with the actual historical functions."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/philosophers_stone'))
from philosophers_stone.philosopher_utils import Config,_compute_wavelet_specs
from phi_extract_common import compute_wavelet_specs_oom_safe
from wavelet import compute_specs
from signal_io import plan_channel


def main():
    cfg=Config()
    cfg.hours_pad=1
    rng=np.random.default_rng(2026102504)
    t=np.arange(24000)/200
    x=15*np.sin(2*np.pi*0.7*t)+5*np.sin(2*np.pi*12*t)+rng.normal(0,1,len(t))
    frame=pd.DataFrame({'c4-m1':x})
    expected=_compute_wavelet_specs(frame,200,cfg).astype(np.float32)
    actual=compute_specs(x,cfg,scale_batch=16)
    short=float(np.max(np.abs(expected-actual)))
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
    # Force multiple fixed-length chunks while retaining the historical padding,
    # CWT dtype conversion, frequency grid, and output interpolation.
    expected_long=compute_wavelet_specs_oom_safe(frame,200,cfg,chunk_len=10000).astype(np.float32)
    actual_long=compute_specs(x,cfg,chunk_len=10000,scale_batch=16)
    long=float(np.max(np.abs(expected_long-actual_long)))
    np.testing.assert_allclose(actual_long,expected_long,rtol=2e-5,atol=2e-6)
    assert plan_channel(['C4','M1'])=={'kind':'derived_c4m1','channels':['C4','M1']}
    assert plan_channel(['C4'])['kind']=='unavailable'
    assert plan_channel(['C4-A2'])['kind']=='unavailable'
    assert plan_channel(['EEG C4-A1'])['kind']=='native_c4m1'
    assert plan_channel(['C3-M2'])['kind']=='unavailable'
    assert plan_channel(['C3-M2'],True)['kind']=='native_c3m2'
    result={'status':'passed','short_max_absolute_error':short,'chunked_max_absolute_error':long,
            'rtol':2e-5,'atol':2e-6,'synthetic_seconds':120,'tested_historical_chunk_length':10000,
            'production_chunk_length':1000000,'production_length_check':'pending_real_record',
            'wavelet_source_sha256':hashlib.sha256((BASE/'wavelet.py').read_bytes()).hexdigest()}
    (BASE/'results/preprocessing_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
