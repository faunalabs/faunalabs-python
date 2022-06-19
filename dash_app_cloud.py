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

from sqlalchemy.sql.operators import op
from sqlalchemy.sql.sqltypes import DateTime
import numpy as np
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float

Base = declarative_base()
from sqlalchemy import update
from scipy import signal
from utils.data import load
import scipy.fftpack
from scipy.signal import find_peaks



external_stylesheets = dbc.themes.DARKLY#'https://codepen.io/chriddyp/pen/bWLwgP.css'

theme = {
    'dark': True,
    'detail': '#00114d',
    'primary': '#000000',
    'secondary': '#c4c4c4',
}

URL = 'mysql+pymysql://sam:whalesrule@35.236.203.47/data'
engine = sqlalchemy.create_engine(URL, pool_size=5,max_overflow=2,pool_timeout=30,pool_recycle=1800)

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
        html.P('Internal Auditing Tool', className='paragraph-lead'),

        html.P('Input your email',className="control_label"),
        dcc.Input(
            id="email_input", type="text", placeholder="Type email", debounce=True,
                        style={ 'width': '200px',
                    'color': theme['secondary'],
                    'background-color': theme['primary'],
                    }),

        html.P('Select Project',className="control_label"),
        dcc.Dropdown(
            id='project_dropdown',
            placeholder='Select dataset to audit',
            style={ 'width': '200px',
                    'color': theme['secondary'],
                    'background-color': theme['primary'],
                    }),

        html.P('Select Deployment',className="control_label"),
        dcc.Dropdown(
            id='deployment_dropdown',
            placeholder='Select deployment to audit',
            style={ 'width': '200px',
                    'color': theme['secondary'],
                    'background-color': theme['primary'],
                    } 
                    ),

        html.P('LEDs',className="control_label"),
        dcc.Checklist(
            id='led_filters',
            options=[
                {'label': '950nm', 'value': 'LED1'},
                {'label': '1050nm', 'value': 'LED2'},
                {'label': '1250nm', 'value': 'LED3'},
                {'label': 'Ambient', 'value': 'AMB'},
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
            value=[],
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
            value=['A_x','A_y','A_z']
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
            value=[]
                ),
        
        html.P('Time Window',className="control_label"),
        #dcc.Input(id="start", type="number", value=0, debounce=True),
        dcc.Input(id="window", type="number", value=5000, debounce=True),


        ],style={'width': '19%', 'display': 'inline-block', 'vertical-align': 'middle'}),
    
    html.Div([
        dash_table.DataTable(
            id='info_table'),
        dcc.Graph(id='graph'),
        dcc.Graph(id='graph_fft'),
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
            {'label': 'Heart Beat', 'value': 'heart'}]
            ),
        html.Div(id='click_data',children=''),
        html.Button('Back', id='back_button',n_clicks=0),
        html.Button('Next', id='forward_button',n_clicks=0)
        ],style={'width': '14%', 'display': 'inline-block', 'vertical-align': 'middle'})

    ])


@app.callback(
    Output('project_dropdown', 'options'),
    Input('email_input', 'value'),
    Input('project_dropdown', 'options'))
def select_project(email,project_list):
    if email == '':
        project_list = []
    else:
        query = 'SELECT * FROM user_info where email="{}"'.format(email)
        user_info = pd.read_sql(query,con=engine)
        query = 'SELECT * FROM user2project where user_id="{}"'.format(user_info['user_id'][0])
        project_list = pd.read_sql(query,con=engine)
        query = 'SELECT * FROM project_info where project_id in ({})'.format(str(project_list['project_id'].to_list())[1:-1])
        project_info = pd.read_sql(query,con=engine)
        project_list = project_info['project_name'].to_list()
    return [{'label': k,'value': k} for k in project_list]

@app.callback(
    Output('deployment_dropdown', 'options'),
    Input('project_dropdown', 'value'),
    Input('deployment_dropdown', 'options'))
def select_deployment(project,deployment_list):
    if project == '':
        deployment_list = []
    else:
        query = 'SELECT * FROM project_info where project_name="{}"'.format(project)
        project_info = pd.read_sql(query,con=engine)
        query = 'SELECT * FROM deployment_meta where project_id="{}"'.format(project_info['project_id'][0])
        deployment_info = pd.read_sql(query,con=engine)
        deployment_list = deployment_info['deployment_name'].to_list()
    return [{'label': k,'value': k} for k in deployment_list]

@app.callback(
    Output('unfiltered_optics', 'data'),
    Output('unfiltered_sensor', 'data'),
    Output('unfiltered_mag', 'data'),
    Output('current_info', 'data'),
    Output('metadata', 'data'),
    Input('deployment_dropdown', 'value'))
def load_data(deployment_name):
    query = 'SELECT * FROM deployment_meta where deployment_name="{}"'.format(deployment_name)
    deployment_meta = pd.read_sql(query,con=engine)
    deployment = deployment_meta['deployment_name'][0] #'ft111_20210511_160612-10'
    optics_df, sensor_df, mag_df, info_df = load(deployment)
    # Load json
    return optics_df.to_json(), sensor_df.to_json(), mag_df.to_json(), info_df.to_json(), deployment_meta.to_json()

@app.callback(
    Output('filtered_optics', 'data'),
    Input('unfiltered_optics', 'data'),
    Input('filtered_sensor', 'data'),
    Input('optics_filters', 'value'),
    Input('forward_button', 'n_clicks'),
    Input('back_button', 'n_clicks'),
    Input('window', 'value'),)
def filter_optics(unfiltered_optics,filtered_sensor,optics_filters,forward_button,back_button,window):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'forward_button' in changed_id:
        new_position = forward_button - back_button
    elif 'back_button' in changed_id:
        new_position = forward_button - back_button
    else:
        new_position = forward_button - back_button

    
    optics_df = pd.read_json(unfiltered_optics)
    #optics_df = optics_df[(optics_df['LED1']<10**led1_max) & (optics_df['LED2']<10**led2_max) & (optics_df['LED3']<10**led3_max) & (optics_df['AMB']<10**ambient_max)]
    
    t_start = new_position*window/1000
    t_end   = t_start + window/1000
    optics_df = optics_df[(optics_df['time'] > t_start) & (optics_df['time'] < t_end)]
    df = optics_df
    #print(changed_id)
    if 'normal' in optics_filters:
        optics_df=(optics_df-optics_df.min())/(optics_df.max()-optics_df.min())
        optics_df['time'] = df['time']
    if 'LP' in optics_filters:
        b, a = signal.butter(10, 20, btype='lowpass', fs=250)
        optics_df['LED1'] = signal.filtfilt(b, a, optics_df['LED1'])
        optics_df['LED2'] = signal.filtfilt(b, a, optics_df['LED2'])
        optics_df['LED3'] = signal.filtfilt(b, a, optics_df['LED3'])
        optics_df['AMB'] = signal.filtfilt(b, a, optics_df['AMB'])
    if 'BP' in optics_filters:
        b, a = signal.butter(10, (0.5, 8), btype='bandpass', fs=250)
        optics_df['LED1'] = signal.filtfilt(b, a, optics_df['LED1'])
        optics_df['LED2'] = signal.filtfilt(b, a, optics_df['LED2'])
        optics_df['LED3'] = signal.filtfilt(b, a, optics_df['LED3'])
        optics_df['AMB'] = signal.filtfilt(b, a, optics_df['AMB'])
    if 'HP' in optics_filters:
        sos =  signal.butter(10, 0.5, 'hp', fs=250, output='sos')
        optics_df['LED1'] = signal.sosfilt(sos, optics_df['LED1'])
        optics_df['LED2'] = signal.sosfilt(sos, optics_df['LED2'])
        optics_df['LED3'] = signal.sosfilt(sos, optics_df['LED3'])
        optics_df['AMB']  = signal.sosfilt(sos, optics_df['AMB'])
    
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
    Input('window', 'value'))
def filter_sensor(unfilterd_sensor,unfiltered_mag,sensor_filters,sensor_selector,forward_button,back_button,window):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'forward_button' in changed_id:
        new_position = forward_button - back_button
    elif 'back_button' in changed_id:
        new_position = forward_button - back_button
    else:
        new_position = forward_button - back_button
    
    t_start = new_position*window/1000
    t_end   = t_start + window/1000

    sensor_df = pd.read_json(unfilterd_sensor)
    mag_df = pd.read_json(unfiltered_mag)
    sensor_df['obda'] = (sensor_df['A_x']*sensor_df['A_x'] + sensor_df['A_y']*sensor_df['A_y'] + sensor_df['A_z']*sensor_df['A_z'])
    sensor_df['obda_diff'] = sensor_df['obda'].diff()
    sensor_df = sensor_df[(sensor_df['time'] > t_start) & (sensor_df['time'] < t_end)]
    df1 = sensor_df
    df2 = mag_df
    #sensor_df, mag_df = cut_noisy_data(sensor_df,mag_df,imu_thresh,imu_window)
    if 'normal' in sensor_filters:
        print('Normalizing Sensor Data')
        sensor_df =(sensor_df-sensor_df.min())/(sensor_df.max()-sensor_df.min())
        sensor_df['time'] = df1['time']
        mag_df =(mag_df-mag_df.min())/(mag_df.max()-mag_df.min())
        mag_df['time'] = df2['time']
    if 'LP' in sensor_filters:
        b, a = signal.butter(10, 20, btype='lowpass', fs=100)
        sensor_df['obda'] = signal.filtfilt(b, a, sensor_df['obda'])
        sensor_df['A_x'] = signal.filtfilt(b, a, sensor_df['A_x'])
        sensor_df['A_y'] = signal.filtfilt(b, a, sensor_df['A_y'])
        sensor_df['A_z'] = signal.filtfilt(b, a, sensor_df['A_z'])
        sensor_df['G_x'] = signal.filtfilt(b, a, sensor_df['G_x'])
        sensor_df['G_y'] = signal.filtfilt(b, a, sensor_df['G_y'])
        sensor_df['G_z'] = signal.filtfilt(b, a, sensor_df['G_z'])
        b, a = signal.butter(10, 20, btype='lowpass', fs=20)
        mag_df['M_x'] = signal.filtfilt(b, a, mag_df['M_x'])
        mag_df['M_y'] = signal.filtfilt(b, a, mag_df['M_y'])
        mag_df['M_z'] = signal.filtfilt(b, a, mag_df['M_z'])
    if 'BP' in sensor_filters:
        b, a = signal.butter(10, (0.5, 8), btype='bandpass', fs=100)
        sensor_df['obda'] = signal.filtfilt(b, a, sensor_df['obda'])
        sensor_df['A_x'] = signal.filtfilt(b, a, sensor_df['A_x'])
        sensor_df['A_y'] = signal.filtfilt(b, a, sensor_df['A_y'])
        sensor_df['A_z'] = signal.filtfilt(b, a, sensor_df['A_z'])
        sensor_df['G_x'] = signal.filtfilt(b, a, sensor_df['G_x'])
        sensor_df['G_y'] = signal.filtfilt(b, a, sensor_df['G_y'])
        sensor_df['G_z'] = signal.filtfilt(b, a, sensor_df['G_z'])
        b, a = signal.butter(10, (0.5, 8), btype='bandpass', fs=20)
        mag_df['M_x'] = signal.filtfilt(b, a, mag_df['M_x'])
        mag_df['M_y'] = signal.filtfilt(b, a, mag_df['M_y'])
        mag_df['M_z'] = signal.filtfilt(b, a, mag_df['M_z'])
    if 'HP' in sensor_filters:
        sos =  signal.butter(10, 0.5, 'hp', fs=100, output='sos')
        sensor_df['obda'] = signal.sosfilt(sos, sensor_df['obda'])
        sensor_df['A_x'] = signal.filtfilt(b, a, sensor_df['A_x'])
        sensor_df['A_y'] = signal.filtfilt(b, a, sensor_df['A_y'])
        sensor_df['A_z'] = signal.filtfilt(b, a, sensor_df['A_z'])
        sensor_df['G_x'] = signal.filtfilt(b, a, sensor_df['G_x'])
        sensor_df['G_y'] = signal.filtfilt(b, a, sensor_df['G_y'])
        sensor_df['G_z'] = signal.filtfilt(b, a, sensor_df['G_z'])
        sos =  signal.butter(10, 0.5, 'hp', fs=20, output='sos')
        mag_df['M_x'] = signal.filtfilt(b, a, mag_df['M_x'])
        mag_df['M_y'] = signal.filtfilt(b, a, mag_df['M_y'])
        mag_df['M_z'] = signal.filtfilt(b, a, mag_df['M_z'])
    
    mag_json = mag_df.to_json()
    sensor_json = sensor_df.to_json()
    return sensor_json, mag_json

@app.callback(
    Output('click_data', 'children'),
    Output('audit', 'data'),
    Input('graph', 'clickData'),
    Input('deployment_dropdown', 'value'),
    Input('email_input', 'value'),
    Input('audit_label', 'value'))
def audit(clickData,deployment_name,email,audit_label):
    query = 'SELECT * FROM deployment_meta where deployment_name="{}"'.format(deployment_name)
    deployment_meta = pd.read_sql(query,con=engine)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print(changed_id)
    if 'clickData' in changed_id:
        query = 'SELECT * FROM user_info where email="{}"'.format(email)
        user_info = pd.read_sql(query,con=engine)
        df = pd.DataFrame({'time':[clickData['points'][0]['x']],
                        'deployment_id':[deployment_meta['deployment_id'][0]],
                        'event':[audit_label],
                        'user_id':[user_info['user_id'][0]]
                        })
        #print(df)
        df.to_sql('audit_data', con=engine,if_exists='append',index=False)

    query = 'SELECT * FROM audit_data where deployment_id="{}"'.format(deployment_meta['deployment_id'][0])
    audit_df = pd.read_sql(query,con=engine)

    return clickData,audit_df.to_json()



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
    Input('audit', 'data'))
def update_graph(forward_button, back_button,window,led_filters,sensor_selector,filtered_optics,filtered_sensor,filtered_mag,audit_json):
    try:
        optics_df = pd.read_json(filtered_optics)
        sensor_df = pd.read_json(filtered_sensor)
        mag_df = pd.read_json(filtered_mag)
        audit_df  = pd.read_json(audit_json)
        sensor_df['obda'] = (sensor_df['A_x']*sensor_df['A_x'] + sensor_df['A_y']*sensor_df['A_y'] + sensor_df['A_z']*sensor_df['A_z'])
        optics_freq = 250
        movement_freq = 100
        mag_freq = 20
        info_freq = 1
        xf = np.linspace(0, optics_freq/2, int(len(optics_df)/2))


        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'forward_button' in changed_id:
            new_position = forward_button - back_button
        elif 'back_button' in changed_id:
            new_position = forward_button - back_button
        else:
            new_position = forward_button - back_button
        
        t_start = new_position*window/1000
        t_end   = t_start + window/1000

        audit_df = audit_df[(audit_df['time'] > t_start) & (audit_df['time'] < t_end)]
        print(audit_df)

        fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
        if 'LED1' in led_filters:
            fig.add_trace(go.Scatter(x=optics_df['time'], y=optics_df['LED1']),
                        row=1, col=1)
        if 'LED2' in led_filters:
            fig.add_trace(go.Scatter(x=optics_df['time'], y=optics_df['LED2']),
                    row=1, col=1)
        if 'LED3' in led_filters:
            fig.add_trace(go.Scatter(x=optics_df['time'], y=optics_df['LED3']),
                    row=1, col=1)
        if 'AMB' in led_filters:
            fig.add_trace(go.Scatter(x=optics_df['time'], y=optics_df['AMB']),
                    row=1, col=1)
        if 'A_x' in sensor_selector:
            fig.add_trace(go.Scatter(x=sensor_df['time'], y=sensor_df['A_x']),
                        row=2, col=1)
        if 'A_y' in sensor_selector:
            fig.add_trace(go.Scatter(x=sensor_df['time'], y=sensor_df['A_y']),
                        row=2, col=1)
        if 'A_z' in sensor_selector:
            fig.add_trace(go.Scatter(x=sensor_df['time'], y=sensor_df['A_z']),
                        row=2, col=1)
        if 'G_x' in sensor_selector:
            fig.add_trace(go.Scatter(x=sensor_df['time'], y=sensor_df['G_x']),
                        row=2, col=1)
        if 'G_y' in sensor_selector:
            fig.add_trace(go.Scatter(x=sensor_df['time'], y=sensor_df['G_y']),
                        row=2, col=1)
        if 'G_z' in sensor_selector:
            fig.add_trace(go.Scatter(x=sensor_df['time'], y=sensor_df['G_z']),
                        row=2, col=1)
        if 'M_x' in sensor_selector:
            fig.add_trace(go.Scatter(x=mag_df['time'], y=mag_df['M_x']),
                        row=2, col=1)
        if 'M_y' in sensor_selector:
            fig.add_trace(go.Scatter(x=mag_df['time'], y=mag_df['M_y']),
                        row=2, col=1)
        if 'M_z' in sensor_selector:
            fig.add_trace(go.Scatter(x=mag_df['time'], y=mag_df['M_z']),
                        row=2, col=1)
        if 'OBDA' in sensor_selector:
            fig.add_trace(go.Scatter(x=sensor_df['time'], y=sensor_df['obda']),
                    row=2, col=1)
        
        try:
            k = 0
            while k < len(audit_df):
                fig.add_vline(x=audit_df['time'][k], row='all',col='all', 
                    annotation_text=audit_df['event'][k], 
                    annotation_position="bottom right")
                k = k + 1
        except Exception as e:
            print(e)
        fig.update_layout(height=800)

        return fig,fig_fft
    except Exception as e:
        print(e)
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