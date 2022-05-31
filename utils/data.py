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

def load(file,optics_freq=250):
    column_names = ['LED2','LED3','LED1','AMB','A_x','A_y','A_z','G_x','G_y','G_z','pitch','roll','heading','depth','temperature','M_x','M_y','M_z',
            'sample_time','voltage','current','power','charge_state','remaining_capacity']
    os.path.join(file)

    print('Parsing File')
    df = pd.read_csv(file,header=None, engine='python').T
    df.columns = column_names
    
    ## Seperate optics data 
    optics_df = df[['LED1','LED2','LED3','AMB']].dropna()
    optics_df['time'] = np.arange(0,len(optics_df)/optics_freq,1/optics_freq)

    ## Seperate movement data
    sensor_df = df[['A_x','A_y','A_z','G_x','G_y','G_z','pitch','roll','heading','depth']].dropna()
    sensor_freq = len(sensor_df)/len(df)*optics_freq
    sensor_df['time'] = np.arange(0,len(sensor_df)/sensor_freq,1/sensor_freq)

    ## Seperate magnetometer data 
    mag_df = df[['M_x','M_y','M_z']].dropna()
    mag_freq = len(mag_df)/len(df)*optics_freq
    mag_df['time'] = np.arange(0,len(mag_df)/mag_freq,1/mag_freq)
    #print(mag_freq)

    info_df = df[['voltage','current','power','charge_state','remaining_capacity']].dropna()
    info_freq = len(info_df)/len(df)*optics_freq
    info_df['time'] = np.arange(0,len(info_df)/info_freq,1/info_freq)

    return optics_df, sensor_df, mag_df, info_df

def save(type,optics_df,sensor_df,mag_df,info_df,audit_df=[],deployment_df=[],location='local',combine = False):
    if combine:
        frames = [optics_df,sensor_df,mag_df,info_df,deployment_df,audit_df]
        df = pd.concat(frames,axis=1)
        if save:
            print('Select directory')
            destination_folder_path = filedialog.askdirectory()
            if type == 'csv':
                df.to_csv(os.path.join(destination_folder_path,'{}_{}.csv'.format(search['term'],search['search'])))
            elif type == 'feather':
                df.to_feather(os.path.join(destination_folder_path,'{}_{}.feather'.format(search['term'],search['search'])))
            elif type == 'mat':
                import scipy.io as sio
                sio.savemat(os.path.join(destination_folder_path,'{}_{}.mat'.format(search['term'],search['search'])), {name: col.values for name, col in df.items()})
            elif type == 'json':
                df.to_json(os.path.join(destination_folder_path,'{}_{}.json'.format(search['term'],search['search'])))
            else:
                print('File type not recognized')
        return df
    
    else:
        return optics_df, sensor_df,mag_df,info_df,deployment_df,audit_df
    if storage == 'SQL':
        print('Uploading to SQL')
        deployment_df.to_sql('deployment_meta', con=engine, if_exists='append', index=False)
        query = 'SELECT * FROM deployment_meta order by deployment_id DESC limit 1'
        deployment_info = pd.read_sql(query,con=engine)
        optics_df['deployment_id'] = deployment_info['deployment_id'][0]
        optics_df.to_sql('optics_data', con=engine, if_exists='append', index=False)
        sensor_df['deployment_id'] = deployment_info['deployment_id'][0]
        sensor_df.to_sql('sensor_data', con=engine, if_exists='append', index=False)
        mag_df['deployment_id'] = deployment_info['deployment_id'][0]
        mag_df.to_sql('mag_data', con=engine, if_exists='append', index=False)
        info_df['deployment_id'] = deployment_info['deployment_id'][0]
        info_df.to_sql('info_data', con=engine, if_exists='append', index=False)
    elif storage == 'GCS':
        print('Uploading to GCS')
        
    elif storage == 'local':
        print('Select directory')
        destination_folder_path = filedialog.askdirectory()
        
        if type == 'csv':
            optics_df.to_csv(os.path.join(destination_folder_path,'{}_{}_optics.csv'.format(search['term'],search['search'])))
            sensor_df.to_csv(os.path.join(destination_folder_path,'{}_{}_sensor.csv'.format(search['term'],search['search'])))
            mag_df.to_csv(os.path.join(destination_folder_path,'{}_{}_mag.csv'.format(search['term'],search['search'])))
            info_df.to_csv(os.path.join(destination_folder_path,'{}_{}_info.csv'.format(search['term'],search['search'])))
            audit_df.to_csv(os.path.join(destination_folder_path,'{}_{}_info.csv'.format(search['term'],search['search'])))
        elif type == 'feather':
            optics_df.to_feather(os.path.join(destination_folder_path,'{}_{}_optics.feather'.format(search['term'],search['search'])))
            sensor_df.to_feather(os.path.join(destination_folder_path,'{}_{}_sensor.feather'.format(search['term'],search['search'])))
            mag_df.to_feather(os.path.join(destination_folder_path,'{}_{}_mag.feather'.format(search['term'],search['search'])))
            info_df.to_feather(os.path.join(destination_folder_path,'{}_{}_info.feather'.format(search['term'],search['search'])))
            audit_df.to_feather(os.path.join(destination_folder_path,'{}_{}_info.feather'.format(search['term'],search['search'])))
        elif type == 'mat':
            import scipy.io as sio
            sio.savemat(os.path.join(destination_folder_path,'{}_{}_optics.mat'.format(search['term'],search['search'])), {name: col.values for name, col in optics_df.items()})
            sio.savemat(os.path.join(destination_folder_path,'{}_{}_sensor.mat'.format(search['term'],search['search'])), {name: col.values for name, col in sensor_df.items()})
            sio.savemat(os.path.join(destination_folder_path,'{}_{}_mag.mat'.format(search['term'],search['search'])), {name: col.values for name, col in mag_df.items()})
            sio.savemat(os.path.join(destination_folder_path,'{}_{}_info.mat'.format(search['term'],search['search'])), {name: col.values for name, col in info_df.items()})
            sio.savemat(os.path.join(destination_folder_path,'{}_{}_audit.mat'.format(search['term'],search['search'])), {name: col.values for name, col in audit_df.items()})
        elif type == 'json':
            optics_df.to_json(os.path.join(destination_folder_path,'{}_{}_optics.json'.format(search['term'],search['search'])))
            sensor_df.to_json(os.path.join(destination_folder_path,'{}_{}_sensor.json'.format(search['term'],search['search'])))
            mag_df.to_json(os.path.join(destination_folder_path,'{}_{}_mag.json'.format(search['term'],search['search'])))
            info_df.to_json(os.path.join(destination_folder_path,'{}_{}_info.json'.format(search['term'],search['search'])))
            audit_df.to_json(os.path.join(destination_folder_path,'{}_{}_audit.json'.format(search['term'],search['search'])))
        else:
            print('File type not recognized')
        print('Saving locally')
    print('Done!')

def transform_data(raw=True,storage='SQL',format='feather'):
    metadata,file = user_input('file')
    if metadata['audit'][0] == 'Yes':
        audit_path = filedialog.askopenfilename()
    optics_freq = metadata['optics_freq'][0]
    root_name = os.path.splitext(file)[0]

    optics_df,sensor_df,mag_df,info_df = parse_data(file,metadata['optics_freq'])

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
                                    'optics_freq':[optics_freq],
                                    'sensor_freq':[sensor_freq],
                                    'mag_freq':[mag_freq],
                                    'info_freq':[info_freq],
                                    })
    save(optics_df,sensor_df,mag_df,info_df,method='GCS')
    return optics_df, sensor_df, mag_df, info_df

def bulk_upload(raw=True,storage='SQL'):
    metadata,path = user_input('bulk_upload')
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if str(name).endswith('.tsv'):
                file = os.path.join(root, name)
                optics_freq = metadata['optics_freq'][0]
                root_name = os.path.splitext(file)[0]
                column_names = ['LED2','LED3','LED1','AMB','A_x','A_y','A_z','G_x','G_y','G_z','pitch','roll','heading','depth','temperature','M_x','M_y','M_z',
                        'sample_time','voltage','current','power','charge_state','remaining_capacity']
                os.path.join(file)

                print('Parsing File: {}'.format(file))
                df = pd.read_csv(file,header=None, engine='python').T
                df.columns = column_names

                ## Seperate optics data 
                optics_df = df[['LED1','LED2','LED3','AMB']].dropna()
                optics_df['time'] = np.arange(0,len(optics_df)/optics_freq,1/optics_freq)

                ## Seperate movement data
                sensor_df = df[['A_x','A_y','A_z','G_x','G_y','G_z','pitch','roll','heading','depth']].dropna()
                sensor_freq = len(sensor_df)/len(df)*optics_freq
                sensor_df['time'] = np.arange(0,len(sensor_df)/sensor_freq,1/sensor_freq)

                ## Seperate magnetometer data 
                mag_df = df[['M_x','M_y','M_z']].dropna()
                mag_freq = len(mag_df)/len(df)*optics_freq
                mag_df['time'] = np.arange(0,len(mag_df)/mag_freq,1/mag_freq)
                #print(mag_freq)

                info_df = df[['voltage','current','power','charge_state','remaining_capacity']].dropna()
                info_freq = len(info_df)/len(df)*optics_freq
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
                                                'optics_freq':[optics_freq],
                                                'sensor_freq':[sensor_freq],
                                                'mag_freq':[mag_freq],
                                                'info_freq':[info_freq],
                                                })
                print('Uploading to SQL')
                deployment_df.to_sql('deployment_meta', con=engine, if_exists='append', index=False)
                query = 'SELECT * FROM deployment_meta order by deployment_id DESC limit 1'
                deployment_info = pd.read_sql(query,con=engine)

                if storage == 'SQL':
                    optics_df['deployment_id'] = deployment_info['deployment_id'][0]
                    optics_df.to_sql('optics_data', con=engine, if_exists='append', index=False)
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
            optics_df,sensor_df,mag_df,info_df = parse_data(local_file,250)
        k = k + 1
    
    return optics_df, sensor_df,mag_df,info_df,deployment_df,audit_df
    