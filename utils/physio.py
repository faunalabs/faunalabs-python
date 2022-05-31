import pandas as pd
import ghostipy as gsp
import numpy as np

def cwt(data,fs,freq_limits=[0.5,8]):

    coefs_cwt, _, f_cwt, t_cwt, _ = gsp.cwt(
        data, fs=fs, freq_limits=freq_limits)

    # will be normalized such that max is 1, so the
    # sampling rate factor can be dropped
    psd_cwt = coefs_cwt.real**2 + coefs_cwt.imag**2
    psd_cwt /= np.max(psd_cwt)

    return t_cwt,f_cwt,psd_cwt


def wsst(data,fs,freq_limits=[0.51,8],vpo=16):

    coefs_wsst, _, f_wsst, t_wsst, _ = gsp.wsst(
            data, fs=fs, freq_limits=freq_limits,
            voices_per_octave=vpo)

    # will be normalized such that max is 1, so the
    # sampling rate factor can be dropped
    psd_wsst = coefs_wsst.real**2 + coefs_wsst.imag**2
    psd_wsst /= np.max(psd_wsst)

    return t_wsst, f_wsst, psd_wsst

