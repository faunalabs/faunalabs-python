import click
import dash
from dash import html,dash_table,dcc
from dash.dependencies import Input,Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
#import functions
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from scipy import signal
from utils.data import load
from utils.physio import wsst, cwt
import scipy.fftpack
from scipy.signal import find_peaks
import plotly.express as px
import os



external_stylesheets = dbc.themes.DARKLY#'https://codepen.io/chriddyp/pen/bWLwgP.css'

theme = {
    'dark': True,
    'detail': '#00114d',
    'primary': '#222222',
    'secondary': '#c4c4c4',
    'text': '#c4c4c4',
}

afe_colors = px.colors.qualitative.Pastel
imu_colors = px.colors.qualitative.Bold
#URL = 'mysql+pymysql://sam:whalesrule@35.236.203.47/data'
#engine = sqlalchemy.create_engine(URL, pool_size=5,max_overflow=2,pool_timeout=30,pool_recycle=1800)

app = dash.Dash(__name__, external_stylesheets=[external_stylesheets])

app.layout = html.Div([ 
    dcc.Store(id='unfiltered_optics'),
    dcc.Store(id='unfiltered_sensor'),
    dcc.Store(id='unfiltered_mag'),
    dcc.Store(id='filtered_optics'),
    dcc.Store(id='filtered_sensor'),
    dcc.Store(id='filtered_mag'),
    dcc.Store(id='current_info'),
    dcc.Store(id='metadata'),
    dcc.Store(id='audit'),
    # This section is the graph that will display on the left part of screen   
    # This is where all filtering options will occur
    html.Div([
        html.H1('FaunaLabs', className='main-title'),
        html.P('Internal Auditing/Vizualisation Tool', className='paragraph-lead'),

        html.P('Select Deployment',className="control_label"),
        dcc.Dropdown(
            id='deployment_dropdown',
            placeholder='Select deployment to audit',
            style={ 'width': '200px',
                    'color': theme['primary'],
                    'background-color': theme['primary'],
                    } 
                    ),

        html.P('LEDs',className="control_label"),
        dcc.Checklist(
            id='led_filters',
            options=[
                {'label': 'LED1', 'value': 'LED1'},
                {'label': 'LED2', 'value': 'LED2'},
                {'label': 'LED3', 'value': 'LED3'},
                {'label': 'AMB', 'value': 'AMB'},
            ],
            value=['LED1','LED2','LED3','AMB']
                ),
        
        html.P('Optics Filters',className="control_label"),
        dcc.Checklist(
            id='optics_filters',
            options=[
                {'label': 'Lowpass', 'value': 'LP'},
                {'label': 'Bandpass', 'value': 'BP'},
                {'label': 'Highpass', 'value': 'HP'},
                {'label': 'Mean Normalized', 'value': 'normal'},
            ],
            value=['normal'],
                ),
        
        html.P('Sensors',className="control_label"),
        dcc.Checklist(
            id='sensor_selector',
            options=[
                {'label': 'Accel X', 'value': 'A_x'},
                {'label': 'Accel Y', 'value': 'A_y'},
                {'label': 'Accel Z', 'value': 'A_z'},
                {'label': 'Gyro X', 'value': 'G_x'},
                {'label': 'Gyro Y', 'value': 'G_y'},
                {'label': 'Gyro Z', 'value': 'G_z'},
                {'label': 'Mag X', 'value': 'M_x'},
                {'label': 'Mag Y', 'value': 'M_y'},
                {'label': 'Mag Z', 'value': 'M_z'},
                {'label': 'OBDA', 'value': 'OBDA'},
            ],
            value=['A_x','G_y']
                ),
        
        html.P('Sensor Filters',className="control_label"),
        dcc.Checklist(
            id='sensor_filters',
            options=[
                {'label': 'Lowpass', 'value': 'LP'},
                {'label': 'Bandpass', 'value': 'BP'},
                {'label': 'Highpass', 'value': 'HP'},
                {'label': 'Mean Normalized', 'value': 'normal'},
            ],
            value=['normal']
                ),
        
        html.P('Time Window',className="control_label"),
        #dcc.Input(id="start", type="number", value=0, debounce=True),
        dcc.Input(id="window", type="number", value=5000, debounce=True),
        html.P('BPM Min/Max',className="control_label"),
        dcc.Input(id="bpm_min", type="number", value=30, debounce=True),
        dcc.Input(id="bpm_max", type="number", value=240, debounce=True),
        html.P('Filter Min/Max',className="control_label"),
        dcc.Input(id="filter_min", type="number", value=0, debounce=True),
        dcc.Input(id="filter_max", type="number", value=10, debounce=True),
        html.P('Filter Order',className="control_label"),
        dcc.Input(id="filter_order", type="number", value=10, debounce=True),

        ],style={'width': '19%', 'display': 'inline-block', 'vertical-align': 'middle'}),
    
    html.Div([
        dash_table.DataTable(
            id='info_table'),
        dcc.Graph(id='graph',figure={
                            'layout': {
                                'plot_bgcolor': theme['primary'],
                                'paper_bgcolor': theme['primary'],
                                'font': {'color': theme['text']}
                                }
                            }
                    ),
        dcc.Graph(id='graph_fft',figure={
                            'layout': {
                                'plot_bgcolor': theme['primary'],
                                'paper_bgcolor': theme['primary'],
                                'font': {'color': theme['text']}
                                }
                            }
                    ),
        ],style={'width': '55%', 'display': 'inline-block', 'vertical-align': 'middle',
                    'color': theme['secondary'],
                    'background-color': theme['primary'],
                    } ),

    html.Div([
        html.P('Select Audit',className="control_label"),
        dcc.Dropdown(
            id='audit_label',
            placeholder='Select label',
            options=[
            {'label': 'Fluke', 'value': 'fluke'},
            {'label': 'Breath', 'value': 'breath'},
            {'label': 'ECG Peak', 'value': 'ecg_peak'},
            {'label': 'PPG Peak', 'value': 'ppg_peak'}],
            style={'color': theme['primary'],
                    'background-color': theme['primary'],}),
        html.Div(id='click_data',children=''),
        html.Button('Back', id='back_button',n_clicks=0),
        html.Button('Next', id='forward_button',n_clicks=0)
        ],style={'width': '14%', 'display': 'inline-block', 'vertical-align': 'middle'})

    ])


@app.callback(
    Output('deployment_dropdown', 'options'),
    Input('deployment_dropdown', 'options'))
def select_deployment(deployment_list):
    from os import listdir
    from os.path import isfile, join
    deployment_list = [f for f in listdir('data') if isfile(join('data', f))]
    return [{'label': k,'value': k} for k in deployment_list]

@app.callback(
    Output('unfiltered_optics', 'data'),
    Output('unfiltered_sensor', 'data'),
    Output('unfiltered_mag', 'data'),
    Output('current_info', 'data'),
    Input('deployment_dropdown', 'value'))
def load_data(deployment):
    if deployment is not None:
        deployment = f'data/{deployment}'
        optics_df, sensor_df, mag_df, info_df = load(deployment)
        # Load json
        print('Data Loaded')
        return optics_df.to_json(), sensor_df.to_json(), mag_df.to_json(), info_df.to_json()

@app.callback(
    Output('filtered_optics', 'data'),
    Input('unfiltered_optics', 'data'),
    Input('optics_filters', 'value'),
    Input('forward_button', 'n_clicks'),
    Input('back_button', 'n_clicks'),
    Input('window', 'value'),
    Input('filter_min', 'value'),
    Input('filter_max', 'value'),
    Input('filter_order', 'value'))
def filter_optics(unfiltered_optics,optics_filters,forward_button,back_button,window,filter_min,filter_max,filter_order):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'forward_button' in changed_id:
        new_position = forward_button - back_button
    elif 'back_button' in changed_id:
        new_position = forward_button - back_button
    else:
        new_position = forward_button - back_button

    
    optics_df = pd.read_json(unfiltered_optics)
    t_start = new_position*window/1000
    t_end   = t_start + window/1000
    optics_df = optics_df[(optics_df['time'] > t_start) & (optics_df['time'] < t_end)]
    df = optics_df
    optics_df['time'] = df['time']
    if 'normal' in optics_filters:
        optics_df=(optics_df-optics_df.min())/(optics_df.max()-optics_df.min())
        optics_df['time'] = df['time']
    if 'LP' in optics_filters:
        if filter_max <= 50:
            sos = signal.butter(filter_order, filter_max, btype='lowpass', fs=250,output='sos')
            print(f'{filter_order}th Order Lowpass with Freq of {filter_max}')
            optics_df['LED1'] = signal.sosfilt(sos, optics_df['LED1'])
            optics_df['LED2'] = signal.sosfilt(sos, optics_df['LED2'])
            optics_df['LED3'] = signal.sosfilt(sos, optics_df['LED3'])
            optics_df['AMB'] = signal.sosfilt(sos, optics_df['AMB'])
        else:
            print('Error with optics lowpass filter')
    if 'BP' in optics_filters:
        if filter_max <= 20 and filter_min > 0 and filter_max > filter_min:
            sos = signal.butter(filter_order, (filter_min, filter_max), btype='bandpass', fs=250,output='sos')
            print(f'{filter_order}th Order Bandpass with Freq of {filter_min, filter_max}')
            optics_df['LED1'] = signal.sosfilt(sos, optics_df['LED1'])
            optics_df['LED2'] = signal.sosfilt(sos, optics_df['LED2'])
            optics_df['LED3'] = signal.sosfilt(sos, optics_df['LED3'])
            optics_df['AMB'] = signal.sosfilt(sos, optics_df['AMB'])
        else:
            print('Error with optics bandpass filter')
    if 'HP' in optics_filters:
        if filter_min > 0 and filter_min < 20:
            sos = signal.butter(filter_order, filter_min, btype='highpass', fs=250,output='sos')
            optics_df['LED1'] = signal.sosfilt(sos, optics_df['LED1'])
            optics_df['LED2'] = signal.sosfilt(sos, optics_df['LED2'])
            optics_df['LED3'] = signal.sosfilt(sos, optics_df['LED3'])
            optics_df['AMB'] = signal.sosfilt(sos, optics_df['AMB'])
            print(f'{filter_order}th Order Highpass with Cuton Freq of {filter_min}')
        else:
            print('Error with optics highpass filter')
    
    #print(optics_df)
    optics_json = optics_df.to_json()
    return optics_json

@app.callback(
    Output('filtered_sensor', 'data'),
    Output('filtered_mag', 'data'),
    Input('unfiltered_sensor', 'data'),
    Input('unfiltered_mag', 'data'),
    Input('sensor_filters', 'value'),
    Input('sensor_selector', 'value'),
    Input('forward_button', 'n_clicks'),
    Input('back_button', 'n_clicks'),
    Input('window', 'value'),
    Input('filter_min', 'value'),
    Input('filter_max', 'value'),
    Input('filter_order', 'value'),)
def filter_sensor(unfilterd_sensor,unfiltered_mag,sensor_filters,sensor_selector,forward_button,back_button,window,filter_min,filter_max,filter_order):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'forward_button' in changed_id:
        new_position = forward_button - back_button
    elif 'back_button' in changed_id:
        new_position = forward_button - back_button
    else:
        new_position = forward_button - back_button
    
    t_start = new_position*window/1000
    t_end   = t_start + window/1000
    try:
        sensor_df = pd.read_json(unfilterd_sensor)
    except Exception as e:
        print(f'Error Reading Sensor Data: {e}')
    try:
        mag_df = pd.read_json(unfiltered_mag)
    except Exception as e:
        print(f'Error Reading Mag Data: {e}')
    sensor_df = sensor_df[(sensor_df['time'] > t_start) & (sensor_df['time'] < t_end)]
    mag_df = mag_df[(mag_df['time'] > t_start) & (mag_df['time'] < t_end)]
    sensor_df['obda'] = (sensor_df['A_x']*sensor_df['A_x'] + sensor_df['A_y']*sensor_df['A_y'] + sensor_df['A_z']*sensor_df['A_z'])
    sensor_df['obda_diff'] = sensor_df['obda'].diff()
    df1 = sensor_df
    df2 = mag_df
    #sensor_df, mag_df = cut_noisy_data(sensor_df,mag_df,imu_thresh,imu_window)

    
    if 'normal' in sensor_filters:
        #print('Normalizing Sensor Data')
        sensor_df =(sensor_df-sensor_df.min())/(sensor_df.max()-sensor_df.min())
        sensor_df['time'] = df1['time']
        mag_df =(mag_df-mag_df.min())/(mag_df.max()-mag_df.min())
        mag_df['time'] = df2['time']
    if 'LP' in sensor_filters :
        if filter_max <= 50:
            b, a = signal.butter(filter_order, filter_max, btype='lowpass', fs=100)
            sensor_df['obda'] = signal.filtfilt(b, a, sensor_df['obda'])
            sensor_df['A_x'] = signal.filtfilt(b, a, sensor_df['A_x'])
            sensor_df['A_y'] = signal.filtfilt(b, a, sensor_df['A_y'])
            sensor_df['A_z'] = signal.filtfilt(b, a, sensor_df['A_z'])
            sensor_df['G_x'] = signal.filtfilt(b, a, sensor_df['G_x'])
            sensor_df['G_y'] = signal.filtfilt(b, a, sensor_df['G_y'])
            sensor_df['G_z'] = signal.filtfilt(b, a, sensor_df['G_z'])
            b, a = signal.butter(filter_order, filter_max, btype='lowpass', fs=100)
            mag_df['M_x'] = signal.filtfilt(b, a, mag_df['M_x'])
            mag_df['M_y'] = signal.filtfilt(b, a, mag_df['M_y'])
            mag_df['M_z'] = signal.filtfilt(b, a, mag_df['M_z'])
        else:
            print('Error with sensor lowpass filter')
    elif 'BP' in sensor_filters:
        if filter_max <= 20 and filter_min > 0 and filter_max > filter_min:
            b, a = signal.butter(filter_order, (filter_min, filter_max), btype='bandpass', fs=100)
            sensor_df['obda'] = signal.filtfilt(b, a, sensor_df['obda'])
            sensor_df['A_x'] = signal.filtfilt(b, a, sensor_df['A_x'])
            sensor_df['A_y'] = signal.filtfilt(b, a, sensor_df['A_y'])
            sensor_df['A_z'] = signal.filtfilt(b, a, sensor_df['A_z'])
            sensor_df['G_x'] = signal.filtfilt(b, a, sensor_df['G_x'])
            sensor_df['G_y'] = signal.filtfilt(b, a, sensor_df['G_y'])
            sensor_df['G_z'] = signal.filtfilt(b, a, sensor_df['G_z'])
            b, a = signal.butter(filter_order, (filter_min, filter_max), btype='bandpass', fs=20)
            mag_df['M_x'] = signal.filtfilt(b, a, mag_df['M_x'])
            mag_df['M_y'] = signal.filtfilt(b, a, mag_df['M_y'])
            mag_df['M_z'] = signal.filtfilt(b, a, mag_df['M_z'])
        else:
            print('Error with sensor bandpass filter')
    elif 'HP' in sensor_filters:
        if filter_min > 0 and filter_min < 20:
            sos =  signal.butter(filter_order, filter_min, 'hp', fs=100)
            sensor_df['obda'] = signal.sosfilt(sos, sensor_df['obda'])
            sensor_df['A_x'] = signal.sosfilt(sos, sensor_df['A_x'])
            sensor_df['A_y'] = signal.sosfilt(sos, sensor_df['A_y'])
            sensor_df['A_z'] = signal.sosfilt(sos, sensor_df['A_z'])
            sensor_df['G_x'] = signal.sosfilt(sos, sensor_df['G_x'])
            sensor_df['G_y'] = signal.sosfilt(sos, sensor_df['G_y'])
            sensor_df['G_z'] = signal.sosfilt(sos, sensor_df['G_z'])
            sos =  signal.butter(filter_order, filter_min, 'hp', fs=20)
            mag_df['M_x'] = signal.sosfilt(sos, mag_df['M_x'])
            mag_df['M_y'] = signal.sosfilt(sos, mag_df['M_y'])
            mag_df['M_z'] = signal.sosfilt(sos, mag_df['M_z'])
        else:
            print('Error with sensor highpass filter')
    mag_json = mag_df.to_json()
    sensor_json = sensor_df.to_json()
    return sensor_json, mag_json

@app.callback(
    Output('audit', 'data'),
    Input('graph', 'clickData'),
    Input('deployment_dropdown', 'value'),
    Input('audit_label', 'value'))
def audit(click_data,deployment_name,audit_label):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    file = f'{deployment_name.split(".")[0]}_audit.txt'

    if os.path.exists(file):
        try:
            audit_df = pd.read_csv(file)
        except Exception as e:
            audit_df = pd.DataFrame()
    else:
        audit_df = pd.DataFrame()
    #print(changed_id)
    if 'clickData' in changed_id:
        #print(click_data)
        
        df = pd.DataFrame({'time':[click_data['points'][0]['x']],
                                    'event':[audit_label]})
        audit_df = pd.concat([audit_df,df],ignore_index=True)
        
        print(f'Saving audit to: {file}')
        audit_df.to_csv(file,index=False)
            
    return audit_df.to_json()



@app.callback(
    Output('graph', 'figure'),
    Output('graph_fft', 'figure'),
    Input('forward_button', 'n_clicks'),
    Input('back_button', 'n_clicks'),
    Input('window', 'value'),
    Input('led_filters', 'value'),
    Input('sensor_selector', 'value'),
    Input('filtered_optics', 'data'),
    Input('filtered_sensor', 'data'),
    Input('filtered_mag', 'data'),
    Input('audit', 'data'),
    Input('bpm_min', 'value'),
    Input('bpm_max', 'value'),
    Input('click_data', 'children'),
    Input('audit_label', 'value'))
def update_graph(forward_button, back_button,window,led_filters,sensor_selector,filtered_optics,filtered_sensor,filtered_mag,audit_json,bpm_min,bpm_max,click_data,audit_label):
    #print(f'LED Filters: {led_filters}')
    if True:#try:
        try:
            #print(f'Filtered Optics {filtered_optics}')
            optics_df = pd.read_json(filtered_optics)
            #print(optics_df.columns)
        except Exception as e:
            print(f'Error Reading Filtered Optics Data: {e}')
            #print(filtered_optics)
        try:
            sensor_df = pd.read_json(filtered_sensor)
            #print(sensor_df.columns)
        except Exception as e:
            print(f'Error Reading Filtered Sensor Data: {e}')
        try:
            mag_df = pd.read_json(filtered_mag)
            #print(mag_df.columns)
        except Exception as e:
            print(f'Error Reading Filtered Mag Data: {e}')
        try:
            audit_df  = pd.read_json(audit_json)
            #print(audit_df)
            #audit_df = audit_df[(audit_df['time']>optics_df.loc[0,'time']) & (audit_df['time']<optics_df.loc[-1,'time'])]
            #print(audit_df)
            #print(audit_df.columns)
        except Exception as e:
            print(f'Error Reading Filtered Audit Data: {e}')
        
        sensor_df['obda'] = (sensor_df['A_x']*sensor_df['A_x'] + sensor_df['A_y']*sensor_df['A_y'] + sensor_df['A_z']*sensor_df['A_z'])
        optics_freq = 250
        sensor_freq = len(sensor_df)/len(optics_df)*optics_freq

        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'forward_button' in changed_id:
            new_position = forward_button - back_button
        elif 'back_button' in changed_id:
            new_position = forward_button - back_button
        else:
            new_position = forward_button - back_button
        
        t_start = new_position*window/1000
        t_end   = t_start + window/1000

        #audit_df = audit_df[(audit_df['time'] > t_start) & (audit_df['time'] < t_end)]
        #print(audit_df)

        fig = make_subplots(rows=1, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
        

        optics_df['combined'] = 0
        sensor_df['combined'] = 0
        if 'LED1' in led_filters:
            fig.add_scatter(x=optics_df['time'], y=optics_df['LED1'],row=1, col=1,name='LED1',marker=dict(color=afe_colors[0]))
            if optics_df['LED1'].isnull().values.any() == False:
                optics_df['combined'] = optics_df['combined'] + optics_df['LED1']
        if 'LED2' in led_filters:
            fig.add_scatter(x=optics_df['time'], y=optics_df['LED2'],row=1, col=1,name='LED2',marker=dict(color=afe_colors[1]))
            if optics_df['LED2'].isnull().values.any() == False:
                optics_df['combined'] = optics_df['combined'] + optics_df['LED2']
        if 'LED3' in led_filters:
            fig.add_scatter(x=optics_df['time'], y=optics_df['LED3'],row=1, col=1,name='LED3',marker=dict(color=afe_colors[2]))
            if optics_df['LED3'].isnull().values.any() == False:
                optics_df['combined'] = optics_df['combined'] + optics_df['LED3']
        if 'AMB' in led_filters:
            fig.add_scatter(x=optics_df['time'], y=optics_df['AMB'],row=1, col=1,name='AMB',marker=dict(color=afe_colors[3]))
            if optics_df['AMB'].isnull().values.any() == False:
                optics_df['combined'] = optics_df['combined'] + optics_df['AMB']
        if 'A_x' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['A_x'],row=1, col=1,name='A_x',marker=dict(color=imu_colors[0]))
            if sensor_df['A_x'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['A_x']
        if 'A_y' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['A_y'],row=1, col=1,name='A_y',marker=dict(color=imu_colors[1]))
            if sensor_df['A_y'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['A_y']
        if 'A_z' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['A_z'],row=1, col=1,name='A_z',marker=dict(color=imu_colors[2]))
            if sensor_df['A_z'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['A_z']
        if 'G_x' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['G_x'],row=1, col=1,name='G_x',marker=dict(color=imu_colors[3]))
            if sensor_df['G_x'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['G_x']
        if 'G_y' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['G_y'],row=1, col=1,name='G_y',marker=dict(color=imu_colors[4]))
            if sensor_df['G_y'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['G_y']
        if 'G_z' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['G_z'],row=1, col=1,name='G_z',marker=dict(color=imu_colors[5]))
            if sensor_df['G_z'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['G_z']
        if 'M_x' in sensor_selector:
            fig.add_scatter(x=mag_df['time'], y=mag_df['M_x'],row=1, col=1,name='M_x',marker=dict(color=imu_colors[6]))
        if 'M_y' in sensor_selector:
            fig.add_scatter(x=mag_df['time'], y=mag_df['M_y'],row=1, col=1,name='M_y',marker=dict(color=imu_colors[7]))
        if 'M_z' in sensor_selector:
            fig.add_scatter(x=mag_df['time'], y=mag_df['M_z'],row=1, col=1,name='M_z',marker=dict(color=imu_colors[8]))
        if 'OBDA' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['obda'],row=1, col=1,name='OBDA',marker=dict(color=imu_colors[9]))
        

        count = []
        if optics_df['combined'].sum() != 0:
            count.append('optics')
            t_wsst,f_wsst,psd_wsst = wsst(optics_df['combined'],optics_freq,freq_limits=[bpm_min/60,bpm_max/60])
            optics_wsst = px.imshow(psd_wsst,aspect="auto",x=t_wsst,y=f_wsst*60,origin="lower").data[0]
        if sensor_df['combined'].sum() != 0:
            count.append('sensor')
            t_wsst,f_wsst,psd_wsst = wsst(sensor_df['combined'],sensor_freq,freq_limits=[bpm_min/60,bpm_max/60])
            sensor_wsst = px.imshow(psd_wsst,aspect="auto",x=t_wsst,y=f_wsst*60,origin="lower").data[0]

        if len(count) == 0:
            fig_fft = {
                "layout": {
                    "xaxis": {
                        "visible": False
                    },
                    "yaxis": {
                        "visible": False
                    },
                    "annotations": [
                        {
                            "text": "",
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {
                                "size": 28
                            }
                        }
                    ]
                    }}
        elif len(count) == 1:
            if 'optics' in count:
                fig_fft = make_subplots(rows=1, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02,
                            subplot_titles=("Optics"))
                fig_fft.add_trace(optics_wsst,1,1)
            if 'sensor' in count:
                fig_fft = make_subplots(rows=1, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02,
                            subplot_titles=("Movement"))
                fig_fft.add_trace(sensor_wsst,1,1)
        elif len(count) == 2:
            fig_fft = make_subplots(rows=1, cols=2,
                            shared_xaxes=True,
                            vertical_spacing=0.02,
                            subplot_titles=("Optics", "Movement"))
            fig_fft.add_trace(optics_wsst,1,1)
            fig_fft.add_trace(sensor_wsst,1,2)
        
        print('Adding Audit Lines')
        k = 0
        while k < len(audit_df):
            if audit_df['time'][k] < optics_df['time'].max() and audit_df['time'][k] > optics_df['time'].min():
                fig.add_vline(x=audit_df['time'][k], row='all',col='all', 
                    annotation_text=audit_df['event'][k], 
                    annotation_position="bottom right")
            k = k + 1

        fig.update_layout(height=500,
                        plot_bgcolor=theme['primary'],
                        paper_bgcolor=theme['primary'],
                        font={'color': theme['text']},
                        xaxis=dict(showgrid=False,zeroline=False),
                        yaxis=dict(showgrid=False,zeroline=False),
                        xaxis_title="Time (seconds)"
                        )

        fig_fft.update_layout(height=500,
                            coloraxis_showscale=False,
                            plot_bgcolor=theme['primary'],
                            paper_bgcolor=theme['primary'],
                            font={'color': theme['text']},
                            xaxis=dict(showgrid=False,zeroline=False),
                            yaxis=dict(showgrid=False,zeroline=False),
                            yaxis_title="Beats/Min",
                            xaxis_title = "Time (s)"
                            )
        
        #fig_fft.update_yaxes(range=[min_fft,max_fft],row=1, col=1)


        return fig, fig_fft
    if False:#except Exception as e:
         print(f'Error: {e}')
         return {
         "layout": {
             "xaxis": {
                 "visible": False
             },
             "yaxis": {
                 "visible": False
             },
             "annotations": [
                 {
                     "text": "Select a Trial",
                     "xref": "paper",
                     "yref": "paper",
                     "showarrow": False,
                     "font": {
                         "size": 28
                     }
                 }
             ]
             }}






@app.callback(
    Output('info_table', 'data'),
    Output('info_table', 'columns'),
    Input('metadata', 'data'))
def metadata(metadata_json):
    df = pd.read_json(metadata_json)
    df_1 = df[['deployment_id','tag_id','project_id','species','animal_id']]
    return df_1.to_dict('records'), [{"name": i, "id": i} for i in df_1.columns],




if __name__ == '__main__':
    app.run_server(debug=True)