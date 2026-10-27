"""Numerical parity against exact submitted preprocessing, without STAGES scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--entry-source',type=Path,required=True)
    parser.add_argument('--upstream-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.upstream_source/'src'))
    sys.path.insert(0,str(args.entry_source))
    from philosophers_stone.philosopher_utils import Config
    from phi_challenge.embed import compute_wavelet_specs_oom_safe
    from reconstruction_wavelet import compute_specs
    cfg=Config();cfg.hours_pad=1
    rng=np.random.default_rng(2026102504)
    t=np.arange(24000)/200
    x=15*np.sin(2*np.pi*.7*t)+5*np.sin(2*np.pi*12*t)+rng.normal(0,1,len(t))
    f=pd.DataFrame({cfg.channel:x})
    checks=[]
    for name,chunk in [('single_chunk',1000000),('fixed_multiple_chunks',10000)]:
        expected=compute_wavelet_specs_oom_safe(f,200,cfg,chunk_len=chunk).astype(np.float32)
        actual=compute_specs(x,cfg,chunk_len=chunk,scale_batch=16)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
        checks.append(dict(name=name,maximum_absolute_error=float(np.max(np.abs(expected-actual))),
                           shape=list(actual.shape),time_chunk_length=chunk))
    result=dict(status='passed',entry_commit='76a36e54251cfce85889a3b2390018ffc23329ee',
                compact_sha256=hashlib.sha256(Path(__file__).with_name('reconstruction_wavelet.py').read_bytes()).hexdigest(),
                checks=checks,rtol=2e-5,atol=2e-6,
                production_time_chunk_length=1000000,frequency_scale_batch=16,
                production_full_length_reference_not_run=True)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
