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


### Source Code
- load.py: Contains functions to load .tsv files from tag
- physio.py: Contains wavelet analysis tools


## Progress
- [x] Basic Auditing and Visualization App
- [x] Physio Analysis Functions (cwt, wsst)
- [x] Basic Filtering Functions
- [ ] Edge Impulse Integration
- [ ] Cloud Storage/SQL Integration
- [ ] Dockerization
- [ ] Hosting
