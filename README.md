# BT4013

Prerequiste
Python 3.8
Anaconda (for setting up of VE)

## Installation guide
**Setting up the virtual environment**
- Create conda environment `conda create -n bt4013 python=3.8`
- Activate environment `conda activate bt4013`
- Install dependencies `pip install -r requirements.txt`
- Execute the main python file `python main.py`

## Running XGBoost Model
- `python xgb_model.py`

## Running ARIMA Model
Due to the large size of saved ARIMA model, The saved models will not be in this repository. As such, there is a need to train and fit models before running the ARIMA models for forecasting and evaluation <br/><br/>
**Fit ARIMA models**
- `python ARIMA.py`
- Models and in-sample predictions will be saved under `ARIMA` folder which will be accessed when performing evaluation of strategies

**Evaluate ARIMA models on trading system**
-Set `strategy` to `arima` in `main.py`
-Run `main.py`
