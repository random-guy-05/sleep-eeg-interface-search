"""Parallel execution of unchanged independent CWT chunks, for remote RAM only."""
from __future__ import annotations
import gc,os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from scipy.interpolate import interp1d

_POOL=None

def _chunk(job):
 from reconstruction_wavelet import compact_chunk
 x,cfg,long=job
 return compact_chunk(x,cfg,scale_batch=128,raw_float32=long)

def compute_specs(eeg_uv_200hz,cfg,*,chunk_len=1_000_000,scale_batch=16):
 from philosophers_stone.philosopher_utils import _make_frequency_grid
 from philosophers_stone.preprocessing_and_spectrograms import pad_spectrogram
 global _POOL
 if _POOL is None:_POOL=ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'))
 x=np.asarray(eeg_uv_200hz,dtype=np.float64)[::2]
 if len(x)%100:raise ValueError('Recording length must be an integer number of seconds')
 long=len(x)>chunk_len;jobs=[]
 for start in range(0,len(x),chunk_len):
  piece=x[start:start+chunk_len]
  if long and len(piece)<chunk_len:piece=np.pad(piece,(0,chunk_len-len(piece)))
  jobs.append((piece,cfg,long))
 pieces=list(_POOL.map(_chunk,jobs));frequencies=pieces[-1][1];wx=np.concatenate([p[0]for p in pieces],axis=0)[:len(x)//100]
 if np.isnan(wx).mean()>.1:raise ValueError('More than 10% CWT samples are NaN')
 wx=np.nan_to_num(wx,nan=0);spec=interp1d(frequencies,wx,axis=1,kind='linear',bounds_error=False,fill_value='extrapolate')(_make_frequency_grid(cfg.n_freqs));spec=pad_spectrogram(spec,cfg.fs_time,hours_pad=cfg.hours_pad)
 if not np.isfinite(spec).all():raise ValueError('Nonfinite spectrogram')
 return np.asarray(spec,dtype=np.float32)

def close_pool():
 global _POOL
 if _POOL is not None:_POOL.shutdown();_POOL=None
