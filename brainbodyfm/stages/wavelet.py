"""Bound memory while preserving the historical fixed-chunk CWT input.

The upstream inference uses the ordinary CWT amplitude returned by ssq_cwt,
not the synchrosqueezed transform. Compute the same CWT scales in small groups
and retain only the 1-Hz time samples used by upstream interpolation. This
avoids storing both full-resolution transforms for an entire night.
"""
from __future__ import annotations
import gc
import numpy as np
from scipy.interpolate import interp1d
from ssqueezepy import cwt, Wavelet
from ssqueezepy.utils import process_scales
from ssqueezepy.ssqueezing import _compute_associated_frequencies


def compact_chunk(x, cfg, *, scale_batch=16, raw_float32=False):
    wavelet = Wavelet((cfg.wavelet_name, {'gamma':cfg.wavelet_gamma,'beta':cfg.wavelet_beta}),N=400)
    scales, scaletype, _, _ = process_scales('log-piecewise',len(x),wavelet,nv=cfg.nv,get_params=True)
    scales=np.asarray(scales).reshape(-1)
    n_seconds=len(x)//100
    if len(x)%100:
        raise ValueError('Fractional-second input requires exact global interpolation support')
    retained=np.empty((n_seconds,len(scales)),dtype=np.float64)
    for start in range(0,len(scales),scale_batch):
        stop=min(start+scale_batch,len(scales))
        # Avoid an unsupported one-scale final group in process_scales.
        if stop-start==1:
            group=scales[start-1:stop]
            wx,_=cwt(x,wavelet,scales=group,fs=100,nv=cfg.nv,l1_norm=True,
                     derivative=False,padtype='reflect',vectorized=True,astensor=False)
            values=np.abs(wx[-1:,::100]).T
        else:
            wx,_=cwt(x,wavelet,scales=scales[start:stop],fs=100,nv=cfg.nv,l1_norm=True,
                     derivative=False,padtype='reflect',vectorized=True,astensor=False)
            values=np.abs(wx[:,::100]).T
        retained[:,start:stop]=values.astype(np.float32) if raw_float32 else values
        del wx,values
    frequencies=_compute_associated_frequencies(scales,len(x),wavelet,scaletype,'peak',was_padded=True,dt=0.01)
    return retained[:,::-1], frequencies


def compute_specs(eeg_uv_200hz, cfg, *, chunk_len=1_000_000, scale_batch=16):
    from philosophers_stone.philosopher_utils import _make_frequency_grid
    from philosophers_stone.preprocessing_and_spectrograms import pad_spectrogram
    x=np.asarray(eeg_uv_200hz,dtype=np.float64)[::2]
    if len(x)%100:
        raise ValueError('Recording length must be an integer number of seconds')
    long=len(x)>chunk_len
    pieces=[]
    frequencies=None
    for start in range(0,len(x),chunk_len):
        piece=x[start:start+chunk_len]
        if long and len(piece)<chunk_len:
            piece=np.pad(piece,(0,chunk_len-len(piece)))
        wx, frequencies=compact_chunk(piece,cfg,scale_batch=scale_batch,raw_float32=long)
        pieces.append(wx)
        gc.collect()
    wx=np.concatenate(pieces,axis=0)[:len(x)//100]
    if np.isnan(wx).mean()>0.1:
        raise ValueError('More than 10% CWT samples are NaN')
    wx=np.nan_to_num(wx,nan=0)
    spec=interp1d(frequencies,wx,axis=1,kind='linear',bounds_error=False,fill_value='extrapolate')(_make_frequency_grid(cfg.n_freqs))
    spec=pad_spectrogram(spec,cfg.fs_time,hours_pad=cfg.hours_pad)
    if not np.isfinite(spec).all():
        raise ValueError('Nonfinite spectrogram')
    return np.asarray(spec,dtype=np.float32)
