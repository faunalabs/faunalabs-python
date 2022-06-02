# Faunalabs: Python Library

This repo contains python code to process data from FaunaLabs products. 

## Usage

### Installation

- Clone this repository
- Install required packages 
`pip install -r requirements`


### Dash App: Auditing + Visualization

#### Dash App Usage
- Place data in repos 'data' directory
- Start app `python dash_app_local.py`
- Navigate to http://127.0.0.1:8050/ 
- Audits will be outputed to repo root directory as filename_audit.txt

![image](https://user-images.githubusercontent.com/28448427/171292706-b423268c-7970-4b66-beab-eacc1e876b38.png)

#### HR Metrics 
Much of these metrics are generated with the HeartPy library 
- beats per minute, BPM
- interbeat interval, IBI
- standard deviation if intervals between adjacent beats, SDNN
- standard deviation of successive differences between adjacent R-R intervals, SDSD
- root mean square of successive differences between adjacend R-R intervals, RMSSD
- proportion of differences between R-R intervals greater than 20ms, 50ms, pNN20, pNN50
- median absolute deviation, MAD
- Poincare analysis (SD1, SD2, S, SD1/SD2)

### Source Code
- load.py: Contains functions to load .tsv files from tag
- physio.py: Contains wavelet analysis tools


## Progress
- [x] Basic Auditing and Visualization App
- [x] Physio Analysis Functions (cwt, wsst)
- [x] Basic Filtering Functions
- [x] Support HeartPy metrics 
- [ ] Edge Impulse Integration
- [ ] Cloud Storage/SQL Integration
- [ ] Dockerization
- [ ] Hosting
