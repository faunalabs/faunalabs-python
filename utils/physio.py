from tracemalloc import start
import pandas as pd
import ghostipy as gsp
import numpy as np
from heartpy.datautils import rolling_mean
import heartpy as hp

def cwt(data,fs,freq_limits=[0.5,8]):

    coefs_cwt, _, f_cwt, t_cwt, _ = gsp.cwt(
        data, fs=fs, freq_limits=freq_limits)

    # will be normalized such that max is 1, so the
    # sampling rate factor can be dropped
    psd_cwt = coefs_cwt.real**2 + coefs_cwt.imag**2
    psd_cwt /= np.max(psd_cwt)

    return t_cwt,f_cwt,psd_cwt


def wsst(data,fs,freq_limits=[0.51,8],vpo=16):

    if freq_limits[0] < .5097:
        freq_limits[0] = .51

    coefs_wsst, _, f_wsst, t_wsst, _ = gsp.wsst(
            data, fs=fs, freq_limits=freq_limits,
            voices_per_octave=vpo)

    # will be normalized such that max is 1, so the
    # sampling rate factor can be dropped
    psd_wsst = coefs_wsst.real**2 + coefs_wsst.imag**2
    psd_wsst /= np.max(psd_wsst)

    return t_wsst, f_wsst, psd_wsst

def find_peaks(data,sample_rate=250,start_t=0):
    wd, measures = hp.process(data, sample_rate)
    y = wd['ybeat']
    x = [(peak/sample_rate)+start_t for peak in wd['peaklist']]
    return x, y

def physio_summary_all(data,sample_rate=250):
    working_data, measures = hp.process(data, sample_rate)
    return measures

def physio_summary_segmentwise(data,sample_rate=250,segment_width = 10, segment_overlap = 0.25):
    working_data, measures = hp.process_segmentwise(data, sample_rate=sample_rate, segment_width = segment_width, segment_overlap = segment_overlap)
    return measures

def physio_summary(afe_df=pd.DataFrame(),sensor_df=pd.DataFrame(),sample_rate=250,segment_width=10,explicit=False,max_bpm=300):
    import math
    
    df = pd.DataFrame(columns=['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate'])

    
    window_df = afe_df.reset_index(drop=True)
    channels = window_df.columns
    window = len(window_df)/sample_rate
    for channel in channels:
        try:
            summary = physio_summary_all(window_df[channel], sample_rate)
            df2 = pd.DataFrame(summary,index=[f'{channel}'])
            df = pd.concat([df,df2])
        except Exception as e:
            if explicit==True:
                print(e)
                df4 = pd.DataFrame(columns=['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate'],index=[channel])
                df = pd.concat([df,df4])

    
    
    window_df = sensor_df.reset_index(drop=True)
    #window_df.dropna(how='all', axis=1, inplace=True)
    channels = window_df.columns
    sample_rate = sample_rate*(len(sensor_df)/len(afe_df))
    for channel in channels:
        try:
            summary = physio_summary_all(window_df[channel], sample_rate)
            print(summary)
            df2 = pd.DataFrame(summary,index=[f'{channel}'])
            df = pd.concat([df,df2])
        except Exception as e:
            if explicit==True:
                print(e)
                df4 = pd.DataFrame(columns=['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate'],index=[channel])
                df = pd.concat([df,df4])

    return df

def plot(data,sample_rate=250):
    working_data, measures = hp.process(data, sample_rate)
    hp.plotter(working_data, measures)