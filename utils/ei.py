# FaunaTag EdgeImpulse Ingester: Python 
#   written by Sam Kelly, 16 July 2021

import json
import time, hmac, hashlib
import sys
import numpy as np
import requests
import pandas as pd
import os 
import math
import tqdm
import data
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'key.json'

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def upload(data_file,audit_file,api_key,hmac_key,window=0.25,device_id='',device_type='2xx'):
    
    afe_df, sensor_df, mag_df, info_df = data.load(data_file)
    audit_df = pd.read_csv(audit_file)

    afe_interval = afe_df['time'][89]-afe_df['time'][88]
    sensor_interval = sensor_df['time'][34]-sensor_df['time'][33]

    if afe_interval != sensor_interval:
        print('Frequency mismatch, using highest common denominator')
        afe_freq = round(1/afe_interval)
        sensor_freq = round(1/sensor_interval)
        print(sensor_freq,afe_freq)
        hcf = math.gcd(sensor_freq, afe_freq)
        afe_df = afe_df.iloc[::int(afe_freq/hcf), :].reset_index(drop=True)
        sensor_df = sensor_df.iloc[::int(sensor_freq/hcf), :].reset_index(drop=True)
        afe_interval = afe_df['time'][89]-afe_df['time'][88]
        sensor_interval = sensor_df['time'][34]-sensor_df['time'][33]

    if afe_interval != sensor_interval:
        sys.exit('Mismatch error with afe and sensor data frequency. Decimation failed.')
    
    try:
        df = afe_df.merge(sensor_df, on=['deployment_id','time'],how='inner')
        df = df[['time','deployment_id','AFE1','AFE2','AFE3','AFE4','A_x','A_y','A_z','G_x','G_y','G_z','depth']]

    ## if not SQL integrated
    except Exception as e:
        df = afe_df.merge(sensor_df, on=['time'],how='inner')
        df = df[['time','AFE1','AFE2','AFE3','AFE4','A_x','A_y','A_z','G_x','G_y','G_z','depth']]

    
    # empty signature (all zeros). HS256 gives 32 byte signature, and we encode in hex, so we need 64 characters here
    emptySignature = ''.join(['0'] * 64)

    # Looping through the audit line-by-line and uploading them to edge impulse
    k = 0 
    from tqdm import tqdm
    for i in tqdm(range(len(audit_df))):
        if audit_df['duration'][k] == 0:
            df = df[df['time'] >= audit_df['time'][k] - window/2]
            df = df[df['time'] <= audit_df['time'][k] + window/2]
        else:
            df = df[df['time'] >= audit_df['time'][k]]
            df = df[df['time'] <= audit_df['time'][k] + audit_df['duration'][k]]

        data = {
            "protected": {
                "ver": "v1",
                "alg": "HS256",
                "iat": time.time() # epoch time, seconds since 1970
            },
            "signature": emptySignature,
            "payload": {
                "device_name": device_id,
                "device_type": device_type,
                "interval_ms": int(sensor_interval*1000),
                "sensors": [
                    { "name": "LED1", "units": "lx" },
                    { "name": "LED2", "units": "lx" },
                    { "name": "LED3", "units": "lx" },
                    { "name": "AMB", "units": "lx" },
                    { "name": "A_x", "units": "m/s2" },
                    { "name": "A_y", "units": "m/s2" },
                    { "name": "A_z", "units": "m/s2" },
                    { "name": "G_x", "units": "m/s" },
                    { "name": "G_y", "units": "m/s" },
                    { "name": "G_z", "units": "m/s" },
                    { "name": "depth", "units": "m" },
                ],
                "values": np.transpose([
                    df['AFE1'].to_list(),
                    df['AFE2'].to_list(),
                    df['AFE3'].to_list(),
                    df['AFE4'].to_list(),
                    pd.to_numeric(df['A_x']).to_list(),
                    pd.to_numeric(df['A_y']).to_list(),
                    pd.to_numeric(df['A_z']).to_list(),
                    pd.to_numeric(df['G_x']).to_list(),
                    pd.to_numeric(df['G_y']).to_list(),
                    pd.to_numeric(df['G_z']).to_list(),
                    pd.to_numeric(df['depth']).to_list(),
                ]).tolist()
            }
        }
        #print(data['payload']['values'])
        # encode in JSON
        encoded = json.dumps(data, default=convert)

        # sign message
        signature = hmac.new(bytes(hmac_key, 'utf-8'), msg = encoded.encode('utf-8'), digestmod = hashlib.sha256).hexdigest()

        # set the signature again in the message, and encode again
        data['signature'] = signature
        encoded = json.dumps(data, default=convert)

    
        # and upload the file
        res = requests.post(url='https://ingestion.edgeimpulse.com/api/training/data',
                            data=encoded,
                            headers={
                                'Content-Type': 'application/json',
                                'x-file-name': '{}_{}_{}'.format(deployment_df['species'][0],audit_df['event'][k],deployment_df['deployment_id'][0]),
                                'x-label': audit_df['event'][k],
                                'x-api-key': api_key
                            })
        if (res.status_code != 200):
            print('Failed to upload file to Edge Impulse', res.status_code, res.content)

        k = k + 1
    res = requests.post(url="https://studio.edgeimpulse.com/v1/api/31547/rebalance",                
                            headers={
                                'x-api-key': api_key
                            })
    print(res.content)

def upload_raw(afe_df,sensor_df,filename,api_key,hmac_key,label='?'):
    
    afe_interval = afe_df['time'][89]-afe_df['time'][88]
    sensor_interval = sensor_df['time'][34]-sensor_df['time'][33]

    if afe_interval != sensor_interval:
        print('Frequency mismatch, using highest common denominator')
        afe_freq = round(1/afe_interval)
        sensor_freq = round(1/sensor_interval)
        print(sensor_freq,afe_freq)
        hcf = math.gcd(sensor_freq, afe_freq)
        afe_df = afe_df.iloc[::int(afe_freq/hcf), :].reset_index(drop=True)
        sensor_df = sensor_df.iloc[::int(sensor_freq/hcf), :].reset_index(drop=True)
        afe_interval = afe_df['time'][89]-afe_df['time'][88]
        sensor_interval = sensor_df['time'][34]-sensor_df['time'][33]

    if afe_interval != sensor_interval:
        sys.exit('Mismatch error with afe and sensor data frequency. Decimation failed.')
    
    try:
        df = afe_df.merge(sensor_df, on=['time'],how='inner')
        df = df[['time','AFE1','AFE2','AFE3','AFE4','A_x','A_y','A_z','G_x','G_y','G_z','depth']]

    ## if not SQL integrated
    except Exception as e:
        df = afe_df.merge(sensor_df, on=['time'],how='inner')
        df = df[['time','AFE1','AFE2','AFE3','AFE4','A_x','A_y','A_z','G_x','G_y','G_z','depth']]

    # empty signature (all zeros). HS256 gives 32 byte signature, and we encode in hex, so we need 64 characters here
    emptySignature = ''.join(['0'] * 64)

    
    data = {
        "protected": {
            "ver": "v1",
            "alg": "HS256",
            "iat": time.time() # epoch time, seconds since 1970
        },
        "signature": emptySignature,
        "payload": {
            "device_name": 111,
            "device_type": 'FaunaTag v1',
            "interval_ms": int(sensor_interval*1000),
            "sensors": [
                { "name": "LED1", "units": "lx" },
                { "name": "LED2", "units": "lx" },
                { "name": "LED3", "units": "lx" },
                { "name": "LED4", "units": "lx" },
                { "name": "A_x", "units": "m/s2" },
                { "name": "A_y", "units": "m/s2" },
                { "name": "A_z", "units": "m/s2" },
                { "name": "G_x", "units": "m/s" },
                { "name": "G_y", "units": "m/s" },
                { "name": "G_z", "units": "m/s" },
                { "name": "depth", "units": "m" },
            ],
            "values": np.transpose([
                df['AFE1'].to_list(),
                df['AFE2'].to_list(),
                df['AFE3'].to_list(),
                df['AFE4'].to_list(),
                pd.to_numeric(df['A_x']).to_list(),
                pd.to_numeric(df['A_y']).to_list(),
                pd.to_numeric(df['A_z']).to_list(),
                pd.to_numeric(df['G_x']).to_list(),
                pd.to_numeric(df['G_y']).to_list(),
                pd.to_numeric(df['G_z']).to_list(),
                pd.to_numeric(df['depth']).to_list(),
            ]).tolist()
        }
    }
    #print(data['payload']['values'])
    # encode in JSON
    encoded = json.dumps(data, default=convert)

    # sign message
    signature = hmac.new(bytes(hmac_key, 'utf-8'), msg = encoded.encode('utf-8'), digestmod = hashlib.sha256).hexdigest()

    # set the signature again in the message, and encode again
    data['signature'] = signature
    encoded = json.dumps(data, default=convert)


    # and upload the file
    res = requests.post(url='https://ingestion.edgeimpulse.com/api/training/data',
                        data=encoded,
                        headers={
                            'Content-Type': 'application/json',
                            'x-file-name': filename,
                            'x-label': label,
                            'x-api-key': api_key
                        })
    if (res.status_code != 200):
        print('Failed to upload file to Edge Impulse', res.status_code, res.content)

    res = requests.post(url="https://studio.edgeimpulse.com/v1/api/31547/rebalance",                
                            headers={
                                'x-api-key': api_key
                            })
    print(res.content)

def mat_upload(decimated_df,audit_df,epoch,trial_name,label,positive_buffer,negative_buffer):
    device_name = '111'
    species = 'turciops'
    # empty signature (all zeros). HS256 gives 32 byte signature, and we encode in hex, so we need 64 characters here
    emptySignature = ''.join(['0'] * 64)
    interval_ms = 1/round(1/(decimated_df['time_s_E{}_DECIMATED'.format(epoch)][1]-decimated_df['time_s_E{}_DECIMATED'.format(epoch)][0]))*1000

    sample_start = decimated_df.iloc[0]['index']
    sample_end   = decimated_df.iloc[-1]['index']
    audit_df = audit_df[(audit_df['index']>sample_start) & (audit_df['index']<sample_end)].reset_index(drop=True)
    # Looping through the audit line-by-line and uploading them to edge impulse
    k = 0
    pbar = tqdm.tqdm(total=len(audit_df))
    while k < len(audit_df):
        x = audit_df['index'][k]
        while True:
            try:
                audit_time = decimated_df[decimated_df['index']==x].reset_index(drop=True)['time_s_E{}_DECIMATED'.format(epoch)][0]
                break
            except Exception as e:
                x = x + 1
                pass
        start_time = audit_time-negative_buffer/1000
        end_time   = audit_time+positive_buffer/1000
        df = decimated_df[decimated_df['time_s_E{}_DECIMATED'.format(epoch)].between(start_time, end_time)]
        data = {
            "protected": {
                "ver": "v1",
                "alg": "HS256",
                "iat": time.time() # epoch time, seconds since 1970
            },
            "signature": emptySignature,
            "payload": {
                "device_name": device_name,
                "device_type": 'FaunaTag v1',
                "interval_ms": interval_ms,
                "sensors": [
                    { "name": "LED1", "units": "lx" },
                    { "name": "LED2", "units": "lx" },
                    { "name": "LED3", "units": "lx" },
                    { "name": "LED4", "units": "lx" },
                    { "name": "A_x", "units": "m/s2" },
                    { "name": "A_y", "units": "m/s2" },
                    { "name": "A_z", "units": "m/s2" },
                    { "name": "G_x", "units": "m/s" },
                    { "name": "G_y", "units": "m/s" },
                    { "name": "G_z", "units": "m/s" },
                    { "name": "depth", "units": "m" },
                ],
                "values": np.transpose([
                    df['AFE1_E{}_DECIMATED'.format(epoch)].to_list(),
                    df['AFE2_E{}_DECIMATED'.format(epoch)].to_list(),
                    df['AFE3_E{}_DECIMATED'.format(epoch)].to_list(),
                    df['AFE4_E{}_DECIMATED'.format(epoch)].to_list(),
                    pd.to_numeric(df['ax_E{}_DECIMATED'].format(epoch)).to_list(),
                    pd.to_numeric(df['ay_E{}_DECIMATED'].format(epoch)).to_list(),
                    pd.to_numeric(df['az_E{}_DECIMATED'].format(epoch)).to_list(),
                    pd.to_numeric(df['gx_E{}_DECIMATED'].format(epoch)).to_list(),
                    pd.to_numeric(df['gy_E{}_DECIMATED'].format(epoch)).to_list(),
                    pd.to_numeric(df['gz_E{}_DECIMATED'].format(epoch)).to_list(),
                    pd.to_numeric(df['depth_E{}_DECIMATED'].format(epoch)).to_list(),
                ]).tolist()
            }
        }
        # encode in JSON
        encoded = json.dumps(data, default=convert)

        # sign message
        signature = hmac.new(bytes(hmac_key, 'utf-8'), msg = encoded.encode('utf-8'), digestmod = hashlib.sha256).hexdigest()

        # set the signature again in the message, and encode again
        data['signature'] = signature
        encoded = json.dumps(data, default=convert)


        # and upload the file
        res = requests.post(url='https://ingestion.edgeimpulse.com/api/training/data',
                            data=encoded,
                            headers={
                                'Content-Type': 'application/json',
                                'x-file-name': '{}.{}'.format(trial_name,k),
                                'x-label': label,
                                'x-api-key': API_KEY
                            })
        if (res.status_code != 200):
            print('Failed to upload file to Edge Impulse', res.status_code, res.content)
        pbar.update(1)
        k = k + 1
    res = requests.post(url="https://studio.edgeimpulse.com/v1/api/31547/rebalance",                
                            headers={
                                'x-api-key': API_KEY
                            })
    print(res.content)
    pbar.close()