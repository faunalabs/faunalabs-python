from tracemalloc import start
import pandas as pd
import ghostipy as gsp
import numpy as np
from heartpy.datautils import rolling_mean
import heartpy as hp
from scipy import signal
from utils.data import resample

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

import numpy as np
from scipy.integrate import simps
from numpy import trapz

def normalize(data):
    if data.max() - data.min() == 0:
        normalized_data = data*0
    else:
        normalized_data = (data-data.min())/(data.max()-data.min())
    return normalized_data

def auc(data,dx,method='simpson'):
    data = data.add(-1*data.min())
    #print(data.min())
    #print(dx)
    if method == 'simpson':
        # Compute the area using the composite Simpson's rule.
        area = simps(data.to_list(), dx=dx)
    elif method == 'trap':
        area = trapz(data.to_list(), dx=dx)
    else:
        print('Method unknown')
    return area

def ptt(ch1,ch2):
    k = 0
    ptt = []
    while k < len(ch1) and k < len(ch2):
        if k == 0:
            x = min([ch1[k]-ch2[k],ch1[k]-ch2[k+1]],key=abs)
        else:
            try:
                x = min([ch1[k]-ch2[k],ch1[k]-ch2[k-1],ch1[k]-ch2[k+1]],key=abs)
            except Exception as e:
                x = min([ch1[k]-ch2[k],ch1[k]-ch2[k-1]],key=abs)
        ptt.append(x)
        k = k + 1
    df = pd.DataFrame([ch1,ch2,ptt])
    return df.T

def find_all_stats(stats_df,df):
    max_slope = []
    max_slope_mag = []
    max_slope_desc = []
    at = []
    dt = []
    diaarea = []
    sysarea = []
    as_ = []
    fas = []
    ds = []
    fds = []
    pir = []
    pw = []
    hr = []
    k = 0
    while k < len(stats_df):
        try:
            start_t = stats_df['f0_x'][k]
            end_t = stats_df['f1_x'][k]
            peak_t = stats_df['p_x'][k]
            window_df = df[(df['time']>start_t) & (df['time']<end_t)].reset_index(drop=True)
            channel = stats_df['channel'][k]
            max_slope.append(window_df.loc[window_df[f'{channel}_diff_norm'].argmax(),'time'])
            max_slope_mag.append(window_df.loc[window_df[f'{channel}_diff_norm'].argmax(),f'{channel}_norm'])
            at.append(peak_t-start_t)
            dt.append(end_t-peak_t)
            diaarea.append(auc(df[(df['time']>start_t)       & (df['time']<max_slope[-1])][f'{channel}_norm'],df['time'][1]-df['time'][0]))
            sysarea.append(auc(df[(df['time']>max_slope[-1]) & (df['time']<end_t)][f'{channel}_norm'],df['time'][1]-df['time'][0]))
            as_.append((stats_df['p_y'][k]-stats_df['f0_y'][k])/at[-1])
            fas.append((max_slope_mag[-1]-stats_df['f0_y'][k])/(max_slope[-1]-start_t))
            ds.append((stats_df['p_y'][k]-stats_df['f0_y'][k])/at[-1])
            fds.append((max_slope_mag[-1]-stats_df['f0_y'][k])/(max_slope[-1]-start_t))
            hr.append(stats_df['p_x'][k+1]-peak_t)
        except Exception as e:
            pass
            #print(e)
        k = k + 1
    #print(max_slope)
    stats_df['m_asc_x'] = pd.Series(max_slope,dtype='float64')
    stats_df['at'] = pd.Series(at,dtype='float64')
    stats_df['dt'] = pd.Series(dt,dtype='float64')
    stats_df['diaarea'] = pd.Series(diaarea,dtype='float64')
    stats_df['sysarea'] = pd.Series(sysarea,dtype='float64')
    stats_df['as'] = pd.Series(as_,dtype='float64')
    stats_df['fas'] = pd.Series(fas,dtype='float64')
    stats_df['ds'] = pd.Series(ds,dtype='float64')
    stats_df['fds'] = pd.Series(fds,dtype='float64')
    stats_df['hr'] = pd.Series(hr,dtype='float64')
    return stats_df


def heart_stats(df,channels,window_length=10):
    freq = int(1/(df['time'][1]-df['time'][0]))
    heart_df = pd.DataFrame()
    for channel in channels:
        print(f'Generating Heart Stats from {channel}')
        try:
            working_data, measures = hp.process_segmentwise(df[f'{channel}_filtered'].to_list(), sample_rate=50, segment_width = window_length, segment_overlap = 0.95)
            df2 = pd.DataFrame(measures)
            df2 = df2.add_prefix(f'{channel}_')
            heart_df = pd.concat([heart_df,df2],axis=1)
        except Exception as e:
            print(e)
            print(f'{channel} failed to find HR data')


    time = []
    step = 0.5
    k = 0
    while k < len(heart_df):
        time.append(k*step+window_length/2)
        k = k + 1

    heart_df['time'] = time
    heart_df = resample(heart_df,freq)
    return heart_df



def generate_heart_info_all(df,window_length,channels=[],filter_type='bp',filter_cut=[0.2,8],filter_order=8,advanced=True):
    
    time_df = pd.DataFrame()
    stats_df = pd.DataFrame()
    done = False
    k = 0
    while done == False:
        start_t = df['time'][0]+k*window_length
        end_t = df['time'][0]+(k+1)*window_length
        if end_t > df.iloc[-1]['time']:
            end_t = df.iloc[-1]['time']
            done = True
        window = [start_t,end_t] 
        print(f'{int(window[0])}-{int(window[1])} secs...')
        time_df2, stats_df2 = generate_heart_info(df,window,channels,filter_type,filter_cut,filter_order,advanced=True)
        time_df = pd.concat([time_df,time_df2],ignore_index = True)
        stats_df = pd.concat([stats_df,stats_df2],ignore_index = True)
        k = k + 1
    print('Done!')
    time_df['td'] = pd.to_timedelta(time_df['time']-time_df['time'][0],'sec')
    time_df.set_index('td',inplace=True,drop=True)
    if advanced==True:
        heart_df = heart_stats(time_df,channels)
        df = pd.concat([df,time_df,heart_df],axis=1)
    else:
        df = pd.concat([df,time_df],axis=1)
    df = df.loc[:,~df.columns.duplicated()].copy()
    return stats_df,df

def generate_heart_info(df,window,channels=[],filter_type='bp',filter_cut=[0.2,8],filter_order=8,advanced=True):
    
    freq = int(1/(df['time'][1]-df['time'][0]))
    window_df = df[(df['time'] > window[0]) & (df['time'] < window[1])]
    df = window_df.reset_index(drop=True)
    
    if filter_type == 'lp':
        sos = signal.butter(filter_order, filter_cut, btype='lowpass', fs=freq,output='sos')
    elif filter_type == 'bp':
        sos = signal.butter(filter_order, filter_cut, btype='bandpass', fs=freq,output='sos')
    elif filter_type == 'hp':
        sos = signal.butter(filter_order, filter_cut, btype='highpass', fs=freq,output='sos')
    else:
        print('Filter type not known')
    
    if len(channels) == 0:
        print('No channels specified')

    stats_df = pd.DataFrame()

    for channel in channels:
        #print(f'Generating {channel} stats')
        df[f'{channel}_norm'] = normalize(df[f'{channel}'])
        df[f'{channel}_filtered'] = signal.sosfilt(sos, df[f'{channel}_norm'])
        df[f'{channel}_diff'] = np.gradient(df[f'{channel}_filtered'],df[channel][1]-df[channel][0])
        df[f'{channel}_diff_norm'] = normalize(df[f'{channel}_diff'])
        df[f'{channel}_diff2'] = np.gradient(df[f'{channel}_diff'],df[channel][1]-df[channel][0])
        df[f'{channel}_diff2_norm'] = normalize(df[f'{channel}_diff2'])

        """         try:
            f0_t = []
            f0_y = []
            f1_t = []
            f1_y = []
            p_t = []
            p_y = []
            try:
                t, y = find_peaks(-1*df[f'{channel}_filtered'],start_t=df['time'][0])
                f0_t = f0_t + t[0:-1]
                f0_y = f0_y + y[0:-1]
                f1_t = f1_t + t[1:]
                f1_y = f1_y + y[1:]
            except Exception as e:
                print(f'Error generating {channel} feet')
                print(e)
        
            try:
                t, y = find_peaks(df[f'{channel}_filtered'],start_t=df['time'][0])
                p_t = p_t + t
                p_y = p_y + y
                
            except Exception as e:
                print(f'Error generating {channel} peaks')
                print(e)
            
            try:
                channel_df = pd.DataFrame()

                channel_df['channel'] = channel
                channel_df['f0_t'] = pd.Series(f0_t,dtype='float64')
                channel_df['f1_t'] = pd.Series(f1_t,dtype='float64')
                channel_df['f0_y'] = pd.Series(f0_y,dtype='float64')
                channel_df['f1_y'] = pd.Series(f1_y,dtype='float64')
                channel_df['p_t'] =  pd.Series(p_t,dtype='float64')
                channel_df['p_y'] =  pd.Series(p_y,dtype='float64')

                stats_df = pd.concat([stats_df,channel_df],ignore_index = True)
            except Exception as e:
                print(f'Error creating {channel} heart_df')
                print(e)
        except Exception as e:
            pass """

    if 'stats_df' in locals():
        stats_df = find_all_stats(stats_df,df)
    else:
        stats_df = pd.DataFrame()
    
    return df,stats_df
