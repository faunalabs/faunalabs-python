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
from utils.physio import wsst, find_peaks,physio_summary
import scipy.fftpack
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
        
        dcc.Checklist(
            id='ecg_toggle',
            options=[
                {'label': 'ECG Active?', 'value': 'ECG'},
            ],
                ),
        dcc.Checklist(
            id='show_peaks',
            options=[
                {'label': 'Show Peaks', 'value': 'show_peaks'},
            ],
                ),

        html.P('LEDs',className="control_label"),
        dcc.Checklist(
            id='led_filters',
            options=[
                {'label': 'Ambient', 'value': 'AFE1'},
                {'label': '1250nm', 'value': 'AFE2'},
                {'label': '1050nm', 'value': 'AFE3'},
                {'label': '950nm/ECG', 'value': 'AFE4'},
            ],
            value=['AFE1','AFE2','AFE3','AFE4']
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
            value=['normal','LP'],
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
            value=['normal','LP']
                ),
        
        html.P('Time Window',className="control_label"),
        #dcc.Input(id="start", type="number", value=0, debounce=True),
        dcc.Input(id="window", type="number", value=10000, debounce=True),
        html.P('BPM Min/Max',className="control_label"),
        dcc.Input(id="bpm_min", type="number", value=30, debounce=True),
        dcc.Input(id="bpm_max", type="number", value=240, debounce=True),
        html.P('Filter Min/Max',className="control_label"),
        dcc.Input(id="filter_min", type="number", value=0, debounce=True),
        dcc.Input(id="filter_max", type="number", value=10, debounce=True),
        html.P('Filter Order',className="control_label"),
        dcc.Input(id="filter_order", type="number", value=6, debounce=True),

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
        dash_table.DataTable(id='datatable',
            style_data={
                'color': 'white',
                'backgroundColor': theme['primary']
                },
            style_data_conditional=[
                {
                'if': {'row_index': 'odd'},
                'backgroundColor': theme['primary'],
                }
                ],
            style_header={
                'backgroundColor': theme['primary'],
                'color': 'white',
                'fontWeight': 'bold'
                }
            ),
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
        afe_df, sensor_df, mag_df, info_df = load(deployment)
        # Load json
        print('Data Loaded')
        return afe_df.to_json(), sensor_df.to_json(), mag_df.to_json(), info_df.to_json()

@app.callback(
    Output('filtered_optics', 'data'),
    Input('unfiltered_optics', 'data'),
    Input('optics_filters', 'value'),
    Input('forward_button', 'n_clicks'),
    Input('back_button', 'n_clicks'),
    Input('window', 'value'),
    Input('filter_min', 'value'),
    Input('filter_max', 'value'),
    Input('filter_order', 'value'),
    )
def filter_optics(unfiltered_optics,optics_filters,forward_button,back_button,window,filter_min,filter_max,filter_order):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'forward_button' in changed_id:
        new_position = forward_button - back_button
    elif 'back_button' in changed_id:
        new_position = forward_button - back_button
    else:
        new_position = forward_button - back_button

    
    afe_df = pd.read_json(unfiltered_optics)
    t_start = new_position*window/1000
    t_end   = t_start + window/1000
    afe_df = afe_df[(afe_df['time'] > t_start) & (afe_df['time'] < t_end)]
    df = afe_df
    afe_df['time'] = df['time']
    if 'normal' in optics_filters:
        afe_df=(afe_df-afe_df.min())/(afe_df.max()-afe_df.min())
        afe_df['time'] = df['time']
    if 'LP' in optics_filters:
        if filter_max <= 50:
            sos = signal.butter(filter_order, filter_max, btype='lowpass', fs=250,output='sos')
            print(f'{filter_order}th Order Lowpass with Freq of {filter_max}')
            afe_df['AFE1'] = signal.sosfilt(sos, afe_df['AFE1'])
            afe_df['AFE2'] = signal.sosfilt(sos, afe_df['AFE2'])
            afe_df['AFE3'] = signal.sosfilt(sos, afe_df['AFE3'])
            afe_df['AFE4'] = signal.sosfilt(sos, afe_df['AFE4'])
        else:
            print('Error with optics lowpass filter')
    if 'BP' in optics_filters:
        if filter_max <= 20 and filter_min >= 0 and filter_max > filter_min:
            sos = signal.butter(filter_order, (filter_min, filter_max), btype='bandpass', fs=250,output='sos')
            print(f'{filter_order}th Order Bandpass with Freq of {filter_min, filter_max}')
            afe_df['AFE1'] = signal.sosfilt(sos, afe_df['AFE1'])
            afe_df['AFE2'] = signal.sosfilt(sos, afe_df['AFE2'])
            afe_df['AFE3'] = signal.sosfilt(sos, afe_df['AFE3'])
            afe_df['AFE4'] = signal.sosfilt(sos, afe_df['AFE4'])
        else:
            print('Error with optics bandpass filter')
    if 'HP' in optics_filters:
        if filter_min > 0 and filter_min < 20:
            sos = signal.butter(filter_order, filter_min, btype='highpass', fs=250,output='sos')
            afe_df['AFE1'] = signal.sosfilt(sos, afe_df['AFE1'])
            afe_df['AFE2'] = signal.sosfilt(sos, afe_df['AFE2'])
            afe_df['AFE3'] = signal.sosfilt(sos, afe_df['AFE3'])
            afe_df['AFE4'] = signal.sosfilt(sos, afe_df['AFE4'])
            print(f'{filter_order}th Order Highpass with Cuton Freq of {filter_min}')
        else:
            print('Error with optics highpass filter')
    
    #print(afe_df)
    optics_json = afe_df.to_json()
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
    Input('filter_order', 'value'),
    Input('ecg_toggle', 'value'))
def filter_sensor(unfilterd_sensor,unfiltered_mag,sensor_filters,sensor_selector,forward_button,back_button,window,filter_min,filter_max,filter_order,ecg_toggle):
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
        if filter_max <= 20 and filter_min >= 0 and filter_max > filter_min:
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
        if filter_min >= 0:
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
    Output('datatable', 'data'),
    Output('datatable', 'columns'),
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
    Input('audit_label', 'value'),
    Input('ecg_toggle', 'value'),
    Input('show_peaks', 'value'))
def update_graph(forward_button, back_button,window,led_filters,sensor_selector,filtered_optics,filtered_sensor,filtered_mag,audit_json,bpm_min,bpm_max,click_data,audit_label,ecg_toggle,show_peaks):


    if True:
        try:
            afe_df = pd.read_json(filtered_optics)
        except Exception as e:
            print(f'Error Reading Filtered Optics Data: {e}')
        try:
            sensor_df = pd.read_json(filtered_sensor)
        except Exception as e:
            print(f'Error Reading Filtered Sensor Data: {e}')
        try:
            mag_df = pd.read_json(filtered_mag)
        except Exception as e:
            print(f'Error Reading Filtered Mag Data: {e}')
        try:
            audit_df  = pd.read_json(audit_json)
        except Exception as e:
            print(f'Error Reading Filtered Audit Data: {e}')
        
        sensor_df['obda'] = (sensor_df['A_x']*sensor_df['A_x'] + sensor_df['A_y']*sensor_df['A_y'] + sensor_df['A_z']*sensor_df['A_z'])
        afe_freq = 250
        sensor_freq = len(sensor_df)/len(afe_df)*afe_freq

        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'forward_button' in changed_id:
            new_position = forward_button - back_button
        elif 'back_button' in changed_id:
            new_position = forward_button - back_button
        else:
            new_position = forward_button - back_button
        
        t_start = new_position*window/1000
        t_end   = t_start + window/1000

        fig = make_subplots(rows=1, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
        #print(led_filters)
        if show_peaks is not None:
            show_peaks = True
        afe_df['combined'] = 0
        sensor_df['combined'] = 0
        if 'AFE1' in led_filters:
            fig.add_scatter(x=afe_df['time'], y=afe_df['AFE1'],row=1, col=1,name='Ambient',marker=dict(color=afe_colors[0]))
            if afe_df['AFE1'].isnull().values.any() == False:
                afe_df['combined'] = afe_df['combined'] + afe_df['AFE1']
            if show_peaks:
                try:
                    x,y = find_peaks(afe_df['AFE1'].reset_index(drop=True),sample_rate=afe_freq,start_t=t_start)
                    fig.add_scatter(x=x, y=y,row=1, col=1,name='Ambient Peaks',marker=dict(color=afe_colors[0]),mode='markers')
                except Exception as e:
                    pass
            
        if 'AFE2' in led_filters:
            fig.add_scatter(x=afe_df['time'], y=afe_df['AFE2'],row=1, col=1,name='1250nm',marker=dict(color=afe_colors[1]))
            if afe_df['AFE2'].isnull().values.any() == False:
                afe_df['combined'] = afe_df['combined'] + afe_df['AFE2']
            if show_peaks:
                try:
                    x,y = find_peaks(afe_df['AFE2'].reset_index(drop=True),sample_rate=afe_freq,start_t=t_start)
                    fig.add_scatter(x=x, y=y,row=1, col=1,name='1250nm Peaks',marker=dict(color=afe_colors[1]),mode='markers')
                except Exception as e:
                    pass
        if 'AFE3' in led_filters:
            fig.add_scatter(x=afe_df['time'], y=afe_df['AFE3'],row=1, col=1,name='1050nm',marker=dict(color=afe_colors[2]))
            if afe_df['AFE3'].isnull().values.any() == False:
                afe_df['combined'] = afe_df['combined'] + afe_df['AFE3']
            if show_peaks:
                try:
                    x,y = find_peaks(afe_df['AFE3'].reset_index(drop=True),sample_rate=afe_freq,start_t=t_start)
                    fig.add_scatter(x=x, y=y,row=1, col=1,name='1050nm Peaks',marker=dict(color=afe_colors[0]),mode='markers')
                except Exception as e:
                    pass
        if 'AFE4' in led_filters:
            if ecg_toggle is None:
                fig.add_scatter(x=afe_df['time'], y=afe_df['AFE4'],row=1, col=1,name='950nm',marker=dict(color=afe_colors[3]))
                if show_peaks:
                    try:
                        x,y = find_peaks(afe_df['AFE4'].reset_index(drop=True),sample_rate=afe_freq,start_t=t_start)
                        fig.add_scatter(x=x, y=y,row=1, col=1,name='950nm Peaks',marker=dict(color=afe_colors[0]),mode='markers')
                    except Exception as e:
                        pass
            else:
                fig.add_scatter(x=afe_df['time'], y=afe_df['AFE4'],row=1, col=1,name='ECG',marker=dict(color=afe_colors[3]))
                if show_peaks:
                    try:
                        x,y = find_peaks(afe_df['AFE4'].reset_index(drop=True),sample_rate=afe_freq,start_t=t_start)
                        fig.add_scatter(x=x, y=y,row=1, col=1,name='ECG Peaks',marker=dict(color=afe_colors[0]),mode='markers')
                    except Exception as e:
                        pass
            if afe_df['AFE4'].isnull().values.any() == False:
                afe_df['combined'] = afe_df['combined'] + afe_df['AFE4']
            
        if 'A_x' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['A_x'],row=1, col=1,name='A_x',marker=dict(color=imu_colors[0]))
            if sensor_df['A_x'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['A_x']
            if show_peaks:
                try:
                    x,y = find_peaks(afe_df['A_x'].reset_index(drop=True),sample_rate=sensor_freq,start_t=t_start)
                    fig.add_scatter(x=x, y=y,row=1, col=1,name='A_x Peaks',marker=dict(color=afe_colors[0]),mode='markers')
                except Exception as e:
                    pass
        if 'A_y' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['A_y'],row=1, col=1,name='A_y',marker=dict(color=imu_colors[1]))
            if sensor_df['A_y'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['A_y']
            if show_peaks:
                try:
                    x,y = find_peaks(afe_df['A_y'].reset_index(drop=True),sample_rate=sensor_freq,start_t=t_start)
                    fig.add_scatter(x=x, y=y,row=1, col=1,name='A_y Peaks',marker=dict(color=afe_colors[0]),mode='markers')
                except Exception as e:
                    pass
        if 'A_z' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['A_z'],row=1, col=1,name='A_z',marker=dict(color=imu_colors[2]))
            if sensor_df['A_z'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['A_z']
            if show_peaks:
                try:
                    x,y = find_peaks(afe_df['A_z'].reset_index(drop=True),sample_rate=sensor_freq,start_t=t_start)
                    fig.add_scatter(x=x, y=y,row=1, col=1,name='A_z Peaks',marker=dict(color=afe_colors[0]),mode='markers')
                except Exception as e:
                    pass
        if 'G_x' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['G_x'],row=1, col=1,name='G_x',marker=dict(color=imu_colors[3]))
            if sensor_df['G_x'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['G_x']
            if show_peaks:
                try:
                    x,y = find_peaks(afe_df['G_x'].reset_index(drop=True),sample_rate=sensor_freq,start_t=t_start)
                    fig.add_scatter(x=x, y=y,row=1, col=1,name='G_x Peaks',marker=dict(color=afe_colors[0]),mode='markers')
                except Exception as e:
                    pass
        if 'G_y' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['G_y'],row=1, col=1,name='G_y',marker=dict(color=imu_colors[4]))
            if sensor_df['G_y'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['G_y']
            if show_peaks:
                try:
                    x,y = find_peaks(afe_df['G_y'].reset_index(drop=True),sample_rate=sensor_freq,start_t=t_start)
                    fig.add_scatter(x=x, y=y,row=1, col=1,name='G_y Peaks',marker=dict(color=afe_colors[0]),mode='markers')
                except Exception as e:
                    pass
        if 'G_z' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['G_z'],row=1, col=1,name='G_z',marker=dict(color=imu_colors[5]))
            if sensor_df['G_z'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['G_z']
            if show_peaks:
                try:
                    x,y = find_peaks(afe_df['G_z'].reset_index(drop=True),sample_rate=sensor_freq,start_t=t_start)
                    fig.add_scatter(x=x, y=y,row=1, col=1,name='G_z Peaks',marker=dict(color=afe_colors[0]),mode='markers')
                except Exception as e:
                    pass
        if 'M_x' in sensor_selector:
            fig.add_scatter(x=mag_df['time'], y=mag_df['M_x'],row=1, col=1,name='M_x',marker=dict(color=imu_colors[6]))
        if 'M_y' in sensor_selector:
            fig.add_scatter(x=mag_df['time'], y=mag_df['M_y'],row=1, col=1,name='M_y',marker=dict(color=imu_colors[7]))
        if 'M_z' in sensor_selector:
            fig.add_scatter(x=mag_df['time'], y=mag_df['M_z'],row=1, col=1,name='M_z',marker=dict(color=imu_colors[8]))
        if 'OBDA' in sensor_selector:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['obda'],row=1, col=1,name='OBDA',marker=dict(color=imu_colors[9]))
        
        physio_df = physio_summary(afe_df=afe_df.filter(led_filters),sensor_df=sensor_df.filter(sensor_selector),explicit=True) 
        physio_df = physio_df.T
        physio_df = physio_df.astype(np.float64)
        physio_df = physio_df.round(2)
        physio_df.insert(0, '', physio_df.index)
        if 'AFE1' in physio_df.columns:
            physio_df.rename(columns = {'AFE1':'AMB'}, inplace = True)
        if 'AFE2' in physio_df.columns:
            physio_df.rename(columns = {'AFE2':'1250nm'}, inplace = True)
        if 'AFE3' in physio_df.columns:
            physio_df.rename(columns = {'AFE3':'1050nm'}, inplace = True)
        if 'AFE4' in physio_df.columns:
            if ecg_toggle is None:
                physio_df.rename(columns = {'AFE4':'950nm'}, inplace = True)
            else:
                physio_df.rename(columns = {'AFE4':'ECG'}, inplace = True)
        #physio_df['index1'] = physio_df.index
        #print(f'Physio DF: {physio_df}')

        if len(physio_df) !=0:
                physio_data=physio_df.to_dict('records'),
                physio_columns=[{'id': c, 'name': c} for c in physio_df.columns],
            
        else:
            physio_data={},
            physio_columns=[],

            
        #dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])

        count = []
        if afe_df['combined'].sum() != 0:
            count.append('optics')
            t_wsst,f_wsst,psd_wsst = wsst(afe_df['combined'],afe_freq,freq_limits=[bpm_min/60,bpm_max/60])
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
        
        #print('Adding Audit Lines')
        k = 0
        while k < len(audit_df):
            if audit_df['time'][k] < afe_df['time'].max() and audit_df['time'][k] > afe_df['time'].min():
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
        
        physio_table = html.Div([
                                dash_table.DataTable(
                            id='physio_summary',
                            columns=physio_columns,
                            data=physio_data
                        )
                        ])

        #print('Physio')
        #print(physio_columns,physio_data)
        print(physio_df)
        return fig, fig_fft,physio_df.to_dict('records'), [{"name": i, "id": i} for i in physio_df.columns]
    if False:#except Exception as e: 
        print(f'Error: {e}')
        fig = make_subplots(rows=1, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
        fig_fft = make_subplots(rows=1, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
        physio_table = make_subplots(rows=1, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
        fig.update_layout(height=500,
                        plot_bgcolor=theme['primary'],
                        paper_bgcolor=theme['primary'],
                        xaxis=dict(showgrid=False,zeroline=False),
                        yaxis=dict(showgrid=False,zeroline=False),
                        )

        fig_fft.update_layout(height=500,
                            plot_bgcolor=theme['primary'],
                            paper_bgcolor=theme['primary'],
                            xaxis=dict(showgrid=False,zeroline=False),
                            yaxis=dict(showgrid=False,zeroline=False),
                            )
        physio_table.update_layout(height=500,
                            plot_bgcolor=theme['primary'],
                            paper_bgcolor=theme['primary'],
                            xaxis=dict(showgrid=False,zeroline=False),
                            yaxis=dict(showgrid=False,zeroline=False),
                            )
        

        return fig, fig_fft,physio_table





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