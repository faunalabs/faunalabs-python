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
from utils.data import load
from utils.physio import generate_heart_info_all



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
    dcc.Store(id='unfiltered_data'),
    dcc.Store(id='filtered_data'),
    dcc.Store(id='heart_data'),
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
            id='afe_channels',
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
            id='filter_type',
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
            id='sensor_channels',
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
        
        #html.P('Sensor Filters',className="control_label"),
        #dcc.Checklist(
        #    id='sensor_filters',
        #    options=[
        #        {'label': 'Lowpass', 'value': 'LP'},
        #        {'label': 'Bandpass', 'value': 'BP'},
        #        {'label': 'Highpass', 'value': 'HP'},
        #        {'label': 'Mean Normalized', 'value': 'normal'},
        #    ],
        #    value=['normal','LP']
        #        ),
        
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
    Output('unfiltered_data', 'data'),
    Input('deployment_dropdown', 'value'))
def load_data(deployment):
    if deployment is not None:
        deployment = f'data/{deployment}'
        df = load(deployment,decimate_freq=50)
        return df.to_json()


@app.callback(
    Output('filtered_data', 'data'),
    Output('heart_data', 'data'),
    Input('unfiltered_data', 'data'),
    Input('filter_type', 'value'),
    Input('filter_min', 'value'),
    Input('filter_max', 'value'),
    Input('filter_order', 'value'),
    Input('afe_channels', 'value'),
    Input('sensor_channels', 'value')
    )
def filter(unfiltered_data,filter_type,filter_min,filter_max,filter_order,afe_channels,sensor_channels):
    print(f'Filtering Data: {filter_type}, {filter_order}th Order, ({filter_min}, {filter_max})')
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    channels = afe_channels + sensor_channels
    df = pd.read_json(unfiltered_data).reset_index(drop=True)
    #print(df)
    if 'LP' in filter_type:
        stats_df, df = generate_heart_info_all(df,10,channels=channels,filter_type='lp',filter_cut=filter_max,filter_order=filter_order,advanced=True)
    elif 'BP' in filter_type:
        stats_df, df = generate_heart_info_all(df,10,channels=channels,filter_type='bp',filter_cut=[filter_min,filter_max],filter_order=filter_order,advanced=True)
    elif 'HP' in filter_type:
        stats_df, df = generate_heart_info_all(df,10,channels=channels,filter_type='hp',filter_cut=filter_min,filter_order=filter_order,advanced=True)
    else:
        pass
    filtered_json = df.to_json()
    heart_json = stats_df.to_json()
    #print('Filtered Data')
    return filtered_json, heart_json

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
    Input('afe_channels', 'value'),
    Input('sensor_channels', 'value'),
    Input('filtered_data', 'data'),
    Input('heart_data', 'data'),
    Input('audit', 'data'),
    Input('bpm_min', 'value'),
    Input('bpm_max', 'value'),
    Input('click_data', 'children'),
    Input('audit_label', 'value'),
    Input('ecg_toggle', 'value'),
    Input('show_peaks', 'value'))
def update_graph(forward_button, back_button,window,afe_channels,sensor_channels,filtered_data,heart_data,audit_data,bpm_min,bpm_max,click_data,audit_label,ecg_toggle,show_peaks):
    print('Updating Graph')

    if True:
        if True: #try:
            df = pd.read_json(filtered_data)
        if False: #except Exception as e:
            print(f'Error Reading Filtered Data: {e}')
        if True: #try:
            heart_df = pd.read_json(heart_data)
        if False: #except Exception as e:
            print(f'Error Reading Filtered Data: {e}')
        if True: #try:
            audit_df  = pd.read_json(audit_data)
        if False: #except Exception as e:
            print(f'Error Reading Filtered Audit Data: {e}')
        
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
        
        #heart_df = heart_df[(heart_df['time'] > t_start) & (heart_df['time'] < t_end)]
        df = df[(df['time'] > t_start) & (df['time'] < t_end)]
        
        
        afe_df = (df.filter(regex="|".join(['AFE', 'time']), axis=1)).reset_index(drop=True)
        sensor_df = (df.filter(regex="|".join(['G_','M_','obda','A_', 'time']), axis=1)).reset_index(drop=True)
        

        sensor_freq = int(1/(sensor_df['time'][1]-sensor_df['time'][0]))
        afe_freq = int(1/(afe_df['time'][1]-afe_df['time'][0]))



        if show_peaks is not None:
            show_peaks = True
        afe_df['combined'] = 0
        sensor_df['combined'] = 0


        if 'AFE1' in afe_channels:
            fig.add_scatter(x=afe_df['time'], y=afe_df['AFE1_filtered'],row=1, col=1,name='Ambient',marker=dict(color=afe_colors[0]))
            if afe_df['AFE1'].isnull().values.any() == False:
                afe_df['combined'] = afe_df['combined'] + afe_df['AFE1_filtered']
           
        if 'AFE2' in afe_channels:
            fig.add_scatter(x=afe_df['time'], y=afe_df['AFE2_filtered'],row=1, col=1,name='1250nm',marker=dict(color=afe_colors[1]))
            if afe_df['AFE2_filtered'].isnull().values.any() == False:
                afe_df['combined'] = afe_df['combined'] + afe_df['AFE2_filtered']
            
        if 'AFE3' in afe_channels:
            fig.add_scatter(x=afe_df['time'], y=afe_df['AFE3_filtered'],row=1, col=1,name='1050nm',marker=dict(color=afe_colors[2]))
            if afe_df['AFE3_filtered'].isnull().values.any() == False:
                afe_df['combined'] = afe_df['combined'] + afe_df['AFE3_filtered']
            
        if 'AFE4' in afe_channels:
            if ecg_toggle is None:
                afe4_name = 'ECG'
            else:
                afe4_name = '950nm'
            fig.add_scatter(x=afe_df['time'], y=afe_df['AFE4_filtered'],row=1, col=1,name=afe4_name,marker=dict(color=afe_colors[3]))
            
            if afe_df['AFE4_filtered'].isnull().values.any() == False:
                afe_df['combined'] = afe_df['combined'] + afe_df['AFE4_filtered']
            
        if 'A_x' in sensor_channels:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['A_x_filtered'],row=1, col=1,name='A_x',marker=dict(color=imu_colors[0]))
            if sensor_df['A_x_filtered'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['A_x']
            
        if 'A_y' in sensor_channels:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['A_y'],row=1, col=1,name='A_y',marker=dict(color=imu_colors[1]))
            if sensor_df['A_y'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['A_y']
            
        if 'A_z' in sensor_channels:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['A_z'],row=1, col=1,name='A_z',marker=dict(color=imu_colors[2]))
            if sensor_df['A_z'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['A_z']
            
        if 'G_x' in sensor_channels:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['G_x'],row=1, col=1,name='G_x',marker=dict(color=imu_colors[3]))
            if sensor_df['G_x'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['G_x']
            
        if 'G_y' in sensor_channels:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['G_y'],row=1, col=1,name='G_y',marker=dict(color=imu_colors[4]))
            if sensor_df['G_y'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['G_y']
            
        if 'G_z' in sensor_channels:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['G_z'],row=1, col=1,name='G_z',marker=dict(color=imu_colors[5]))
            if sensor_df['G_z'].isnull().values.any() == False:
                sensor_df['combined'] = sensor_df['combined'] + sensor_df['G_z']
            
        if 'M_x' in sensor_channels:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['M_x'],row=1, col=1,name='M_x',marker=dict(color=imu_colors[6]))
        if 'M_y' in sensor_channels:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['M_y'],row=1, col=1,name='M_y',marker=dict(color=imu_colors[7]))
        if 'M_z' in sensor_channels:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['M_z'],row=1, col=1,name='M_z',marker=dict(color=imu_colors[8]))
        if 'OBDA' in sensor_channels:
            fig.add_scatter(x=sensor_df['time'], y=sensor_df['obda'],row=1, col=1,name='OBDA',marker=dict(color=imu_colors[9]))
        
        """ physio_df = physio_summary(afe_df=afe_df.filter(afe_channels),sensor_df=sensor_df.filter(sensor_channels),explicit=True) 
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
                physio_df.rename(columns = {'AFE4':'ECG'}, inplace = True) """
        physio_df = pd.DataFrame()
        if len(physio_df) !=0:
                physio_data=physio_df.to_dict('records'),
                physio_columns=[{'id': c, 'name': c} for c in physio_df.columns],
            
        else:
            physio_data={},
            physio_columns=[],

            
        
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
        print(afe_df)
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





#@app.callback(
#    Output('info_table', 'data'),
#    Output('info_table', 'columns'),
#    Input('metadata', 'data'))
#def metadata(metadata_json):
#    df = pd.read_json(metadata_json)
#    df_1 = df[['deployment_id','tag_id','project_id','species','animal_id']]
#    return df_1.to_dict('records'), [{"name": i, "id": i} for i in df_1.columns],




if __name__ == '__main__':
    app.run_server(debug=True)