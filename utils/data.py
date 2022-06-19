##### FaunaLabs #####
## Author: Sam Kelly

# This code is currently proprietary, further licensing will be decided in the near future

"""
This script is designed to communicate between the SQL database hosted on Google Cloud Platform and your local machine
"""
from __future__ import print_function, unicode_literals
import datetime
import logging
import os
import pandas as pd
import uuid
import time
import datetime
import numpy as np

def upload_audit(audit_df):
    user_id, deployment_id = user_input(type='upload_audit')
    audit_df['user_id'] = user_id
    audit_df['deployment_id'] = deployment_id
    audit_df.to_sql('audit_data', con=engine, if_exists='append', index=False)
    print('Successful audit upload')
    return

def amend_datafile():
    print('This is not yet developed')
    return

def resample(df,freq):
    df['td'] = pd.to_timedelta(df['time']-df['time'][0],'sec')
    df.set_index('td',inplace=True,drop=True)
    df = df.resample(f'{int((1/freq)*1000)}ms').ffill()
    df.index.floor(f'{int((1/freq)*1000)}ms')
    return df

def load(file,reload=False,afe_freq=250,decimate_freq=None,start=0,duration=None):
    
    column_names = ['AFE1','AFE2','AFE3','AFE4','A_x','A_y','A_z','G_x','G_y','G_z','pitch','roll','heading','depth','temperature','M_x','M_y','M_z',
            'sample_time','voltage','current','power','charge_state','remaining_capacity']
    os.path.join(file)

    if reload == False:
        print('Attempting reload existing data')
        try:
            df = pd.read_feather(f'{file.split(".")[0]}.feather')
            print('Data reloaded')
            return df
        except Exception as e:
            print('Reload Failed... Parsing CSV File')
            df = pd.read_csv(file,header=None, engine='python').T
            df.columns = column_names
            print('Data Loaded')
            save(df,file)
    else:
        print('Parsing CSV File')
        df = pd.read_csv(file,header=None, engine='python').T
        df.columns = column_names
        print('Data Loaded')
        save(df,file)
            
    ## Seperate optics data 
    afe_df = df[['AFE1','AFE2','AFE3','AFE4']].dropna()
    afe_df['time'] = np.arange(0,len(afe_df)/afe_freq,1/afe_freq)
    
    start = start*60
    if duration == None:
        duration = afe_df.iloc[-1]['time']
    else:
        duration = duration*60
    
    afe_df = afe_df[(afe_df['time']>start) & (afe_df['time']<(start+duration))].reset_index(drop=True)
    
    ## Seperate movement data
    sensor_df = df[['A_x','A_y','A_z','G_x','G_y','G_z','pitch','roll','heading','depth']].dropna()
    sensor_df['obda'] = (sensor_df['A_x']*sensor_df['A_x'] + sensor_df['A_y']*sensor_df['A_y'] + sensor_df['A_z']*sensor_df['A_z'])
    sensor_df['obda_diff'] = sensor_df['obda'].diff()
    sensor_freq = len(sensor_df)/len(df)*afe_freq
    sensor_df['time'] = np.arange(0,len(sensor_df)/sensor_freq,1/sensor_freq)
    sensor_df = sensor_df[(sensor_df['time']>start) & (sensor_df['time']<(start+duration))].reset_index(drop=True)

    ## Seperate magnetometer data 
    mag_df = df[['M_x','M_y','M_z']].dropna()
    mag_freq = len(mag_df)/len(df)*afe_freq
    mag_df['time'] = np.arange(0,len(mag_df)/mag_freq,1/mag_freq)
    mag_df = mag_df[(mag_df['time']>start) & (mag_df['time']<(start+duration))].reset_index(drop=True)
    #print(mag_freq)

    info_df = df[['voltage','current','power','charge_state','remaining_capacity']].dropna()
    info_freq = len(info_df)/len(df)*afe_freq
    info_df['time'] = np.arange(0,len(info_df)/info_freq,1/info_freq)
    info_df = info_df[(info_df['time']>start) & (info_df['time']<(start+duration))].reset_index(drop=True)
    
    if decimate_freq == None:
        return afe_df, sensor_df, mag_df, info_df
    else:
        df = pd.concat([resample(afe_df,decimate_freq), resample(sensor_df,decimate_freq), resample(mag_df,decimate_freq),resample(info_df,decimate_freq)],axis=1)
        df['time'] = np.arange(0,len(df)/decimate_freq,1/decimate_freq)
        df = df.loc[:,~df.columns.duplicated()].copy()
        return df

def save(df,file,type='feather'):
    if type == 'feather':
        print('Saving feather')
        df.to_feather(f'{file.split(".")[0]}.feather')
    
    
def transform_data(raw=True,storage='SQL',format='feather'):
    metadata,file = user_input('file')
    if metadata['audit'][0] == 'Yes':
        audit_path = filedialog.askopenfilename()
    afe_freq = metadata['afe_freq'][0]
    root_name = os.path.splitext(file)[0]

    afe_df,sensor_df,mag_df,info_df = parse_data(file,metadata['afe_freq'])

    print(metadata)
    deployment_df = pd.DataFrame({'tag_id':[metadata['tag_id'][0]],
                                    'deployment_name':[metadata['deployment_name'][0]],
                                    'project_id':[metadata['project_id'][0]],
                                    'description':[metadata['description'][0]],
                                    'time':[datetime.datetime.strptime(metadata['date_time'][0], '%m-%d-%Y')],
                                    'species':[metadata['species'][0]],
                                    'sex':[metadata['sex'][0]],
                                    'animal_id':[metadata['animal_id'][0]],
                                    'life_stage':[metadata['life_stage'][0]],
                                    'afe_freq':[afe_freq],
                                    'sensor_freq':[sensor_freq],
                                    'mag_freq':[mag_freq],
                                    'info_freq':[info_freq],
                                    })
    save(afe_df,sensor_df,mag_df,info_df,method='GCS')
    return afe_df, sensor_df, mag_df, info_df

def bulk_upload(raw=True,storage='SQL'):
    metadata,path = user_input('bulk_upload')
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if str(name).endswith('.tsv'):
                file = os.path.join(root, name)
                afe_freq = metadata['afe_freq'][0]
                root_name = os.path.splitext(file)[0]
                column_names = ['AFE1','AFE2','AFE3','AFE4','A_x','A_y','A_z','G_x','G_y','G_z','pitch','roll','heading','depth','temperature','M_x','M_y','M_z',
                        'sample_time','voltage','current','power','charge_state','remaining_capacity']
                os.path.join(file)

                print('Parsing File: {}'.format(file))
                df = pd.read_csv(file,header=None, engine='python').T
                df.columns = column_names

                ## Seperate optics data 
                afe_df = df[['AFE3','AFE1','AFE2','AFE4']].dropna()
                afe_df['time'] = np.arange(0,len(afe_df)/afe_freq,1/afe_freq)

                ## Seperate movement data
                sensor_df = df[['A_x','A_y','A_z','G_x','G_y','G_z','pitch','roll','heading','depth']].dropna()
                sensor_freq = len(sensor_df)/len(df)*afe_freq
                sensor_df['time'] = np.arange(0,len(sensor_df)/sensor_freq,1/sensor_freq)

                ## Seperate magnetometer data 
                mag_df = df[['M_x','M_y','M_z']].dropna()
                mag_freq = len(mag_df)/len(df)*afe_freq
                mag_df['time'] = np.arange(0,len(mag_df)/mag_freq,1/mag_freq)
                #print(mag_freq)

                info_df = df[['voltage','current','power','charge_state','remaining_capacity']].dropna()
                info_freq = len(info_df)/len(df)*afe_freq
                info_df['time'] = np.arange(0,len(info_df)/info_freq,1/info_freq)

                #print(metadata)
                deployment_df = pd.DataFrame({'tag_id':[metadata['tag_id'][0]],
                                                'deployment_name':[name],
                                                'project_id':[metadata['project_id'][0]],
                                                'description':[metadata['description'][0]],
                                                'time':[datetime.datetime.strptime(metadata['date_time'][0], '%m/%d/%Y')],
                                                'species':[metadata['species'][0]],
                                                'sex':[metadata['sex'][0]],
                                                'animal_id':[metadata['animal_id'][0]],
                                                'life_stage':[metadata['life_stage'][0]],
                                                'afe_freq':[afe_freq],
                                                'sensor_freq':[sensor_freq],
                                                'mag_freq':[mag_freq],
                                                'info_freq':[info_freq],
                                                })
                print('Uploading to SQL')
                deployment_df.to_sql('deployment_meta', con=engine, if_exists='append', index=False)
                query = 'SELECT * FROM deployment_meta order by deployment_id DESC limit 1'
                deployment_info = pd.read_sql(query,con=engine)

                if storage == 'SQL':
                    afe_df['deployment_id'] = deployment_info['deployment_id'][0]
                    afe_df.to_sql('optics_data', con=engine, if_exists='append', index=False)
                    sensor_df['deployment_id'] = deployment_info['deployment_id'][0]
                    sensor_df.to_sql('sensor_data', con=engine, if_exists='append', index=False)
                    mag_df['deployment_id'] = deployment_info['deployment_id'][0]
                    mag_df.to_sql('mag_data', con=engine, if_exists='append', index=False)
                    info_df['deployment_id'] = deployment_info['deployment_id'][0]
                    info_df.to_sql('info_data', con=engine, if_exists='append', index=False)
                elif storage == 'GCP':
                    print('Not yet complete')
                print('Complete')
    print('Done!')

def download(user_input = True,common_freq=False,type='csv',save=True, search='',term='',combine=False,freq=''):
    while 1:
        questions = [{'type': 'list','name': 'term','message': 'What category are you looking for?','choices': ['deployment_id','project_id','deployment_name','sex','species','animal_id','tag_id','time']},
                    {'type': 'input','name': 'search','message': 'What is the search term'}
                    ]
        if user_input == True:
            search = prompt(questions)
            term = search['term']
            search = search['search']


    
        #query = 'SELECT * FROM info_data where deployment_id in ({})'.format(str(deployment_df['deployment_id'].to_list())[1:-1])
        try:
        #    info_df = pd.read_sql(query,con=engine)
        #    print('Loaded Deployment Info')
            query = 'SELECT * FROM deployment_meta where {}="{}"'.format(term,search)
            deployment_df = pd.read_sql(query,con=engine)
            print('Loaded Deployment Info')
            break
        except Exception as e:
            print(e)
            print('Error: Couldnt find record with these search terms')

    query = 'SELECT * FROM tag_info where tag_id in ({})'.format(str(deployment_df['tag_id'].to_list())[1:-1])
    tag_df = pd.read_sql(query,con=engine)
    deployment_df = deployment_df.merge(tag_df, how='left', on='tag_id')
    print('Loaded Tag Info')

    query = 'SELECT * FROM project_info where project_id in ({})'.format(str(deployment_df['project_id'].to_list())[1:-1])
    project_df = pd.read_sql(query,con=engine)
    deployment_df = deployment_df.merge(project_df, how='left', on='project_id')
    print('Loaded Project Info')

    #query = 'SELECT * FROM testing where tag_id in ({}) ORDER by date DESC limit {}'.format(str(tag_df['tag_id'].to_list())[1:-1], len(tag_df))
    #calibration_df = pd.read_sql(query,con=engine)
    #deployment_df = deployment_df.merge(calibration_df, how='left', on='deployment_id')
    #print('Loaded Calibration Info')

    query = 'SELECT * FROM audit_data where deployment_id in ({})'.format(str(deployment_df['deployment_id'].to_list())[1:-1])
    audit_df = pd.read_sql(query,con=engine)
    print('Loaded Audits')

    ## Download the relevant data objects

    # try to find a feather object first
    
    k = 0 
    while k < len(deployment_df):
        
        bucket = storage_client.bucket('data-faunalabs')
        project_code = deployment_df['project_code'][k]
        tag_id = deployment_df['tag_id'][k]
        deployment_name = deployment_df['deployment_name'][k]
        date = deployment_df['time'][k]
        date = '{}{:02d}{:02d}'.format(date.year,date.month,date.day)
        extension = '.tsv'
        save_directory = '_temp'
        file = '{}{}'.format(deployment_name,extension)
        local_file = '{}/{}'.format(save_directory,file)

        query = '{}/FaunaTag{}/{}/{}'.format(project_code,tag_id,date,file)
        print(query)
        blob = bucket.blob(query)
        if os.path.isdir(save_directory) == False:
            os.mkdir(save_directory)
        blob.download_to_filename(local_file)
        if extension =='.tsv' or '.csv':
            afe_df,sensor_df,mag_df,info_df = parse_data(local_file,250)
        k = k + 1
    
    return afe_df, sensor_df,mag_df,info_df,deployment_df,audit_df
    