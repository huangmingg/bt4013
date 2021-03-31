import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import functools
import os
from xgb_model import predict, transform_data, REQUIRED_COLS
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.risk_models import CovarianceShrinkage
from ta.volume import OnBalanceVolumeIndicator
import pickle
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from pmdarima.arima import auto_arima
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import json


def baseline(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    pos = np.array([1 / 89 for i in settings['markets']])
    return pos, settings


def trend_following(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    periodLonger = 200
    periodShorter = 40
    nMarkets = CLOSE.shape[1]
    # Calculate Simple Moving Average (SMA)
    smaLongerPeriod = np.nansum(CLOSE[-periodLonger:, :], axis=0) / periodLonger
    smaShorterPeriod = np.nansum(CLOSE[-periodShorter:, :], axis=0) / periodShorter

    longEquity = smaShorterPeriod > smaLongerPeriod
    shortEquity = ~longEquity

    pos = np.zeros(nMarkets)
    pos[longEquity] = 1
    pos[shortEquity] = -1

    weights = pos / np.nansum(abs(pos))

    return weights, settings


def sma_504(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets = CLOSE.shape[1]
    df = pd.DataFrame(CLOSE)
    df.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)
    df = df[list(filter(lambda x: x != 'CASH', settings['markets']))]
    df.dropna(axis='columns', inplace=True)
    if df.empty:
        if settings['history']:
            weights = settings['history'][-1]
        else:
            weights = np.array([1 if i == 'CASH' else 0 for i in settings['markets']])
    else:
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)
        ef = EfficientFrontier(mu, S)
        try:
            optimised_weights = ef.max_sharpe()
            weights = np.array([value for key, value in optimised_weights.items()])
            pos = np.zeros(nMarkets)
            for j in range(0, len(weights)):
                pos[settings['markets'].index(df.columns[j])] = weights[j]
            weights = pos
            print("Optimal solution found!")
        except Exception:
            print("No optimal solution, using same weights allocation in the previous timestep")
            if settings['history']:
                weights = settings['history'][-1]
            else:
                weights = np.array([1 if i == 'CASH' else 0 for i in settings['markets']])
        settings['history'].append(weights)
    return weights, settings


def ema_504(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets = CLOSE.shape[1]
    df = pd.DataFrame(CLOSE)
    df.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)
    df = df[list(filter(lambda x: x != 'CASH', settings['markets']))]
    df.dropna(axis='columns', inplace=True)
    if df.empty:
        if settings['history']:
            weights = settings['history'][-1]
        else:
            weights = np.array([1 if i == 'CASH' else 0 for i in settings['markets']])
    else:
        mu = expected_returns.ema_historical_return(df)
        S = risk_models.sample_cov(df)
        ef = EfficientFrontier(mu, S)
        try:
            optimised_weights = ef.max_sharpe()
            weights = np.array([value for key, value in optimised_weights.items()])
            pos = np.zeros(nMarkets)
            for j in range(0, len(weights)):
                pos[settings['markets'].index(df.columns[j])] = weights[j]
            weights = pos
            print("Optimal solution found!")
        except Exception:
            print("No optimal solution, using same weights allocation in the previous timestep")
            if settings['history']:
                weights = settings['history'][-1]
            else:
                weights = np.array([1 if i == 'CASH' else 0 for i in settings['markets']])
        settings['history'].append(weights)
    return weights, settings


def pairs_trade(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets = CLOSE.shape[1]
    markets = settings["markets"]
    # trade 2 pairs, bg-bc and ec-dx
    treshold = 0.

    pos1 = np.zeros(nMarkets)
    pos2 = np.zeros(nMarkets)

    for i in range(0, nMarkets):
        future_name = markets[i]
        close = np.transpose(CLOSE)
        if future_name == 'F_BO':
            fbg_all = close[i][:-50]
        if future_name == 'F_W':
            fbc_all = close[i][:-50]

    hr = (fbg_all / fbc_all).mean()
    for i in range(0, nMarkets):
        future_name = markets[i]
        close = np.transpose(CLOSE)
        if future_name == 'F_BO':
            fbg_price = close[i][-1]
            fbg_i = i
        if future_name == 'F_W':
            fbc_price = close[i][-1]
            fbc_i = i

    if fbg_price / fbc_price > hr + treshold:  # historical mean
        # short fbg long fbc
        # print("short fbg long fbc")

        pos2[fbc_i] = 0.5
        pos2[fbg_i] = -0.5

    elif fbg_price / fbc_price < hr - treshold:
        # long fbg short fbc
        # print("long fbg short fbc")
        pos2[fbc_i] = -0.5
        pos2[fbg_i] = 0.5
    else:
        pos2[fbc_i] = 0
        pos2[fbg_i] = 0

    pos = pos2
    return pos, settings


def on_balance_volume(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    df = pd.DataFrame(CLOSE)
    df1 = pd.DataFrame(VOL)
    df.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)
    df1.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)

    columns = list(df.columns)
    obv = {}
    for x in columns:
        if x == 'CASH':
            obv[x] = 0
        else:
            obv[x] = OnBalanceVolumeIndicator(df[x], df1[x]).on_balance_volume().iloc[-1]

    pos = list(obv.values())
    # long if obv > 0, short if obv < 0
    pos = list(map(lambda x: 1 if x > 0 else -1 if x < -1 else 0, pos))
    pos = np.array(pos)
    weights = pos / np.nansum(abs(pos))
    return weights, settings


def mean_reversion(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    df = pd.DataFrame(CLOSE)
    df1 = pd.DataFrame(HIGH)
    df2 = pd.DataFrame(LOW)
    df.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)
    df1.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)
    df2.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)

    columns = list(df.columns)
    rsi2 = {}
    atr22 = {}
    sma200 = {}

    for x in columns:
        if x == 'CASH':
            rsi2[x] = 0
            sma200[x] = 0

        else:
            rsi2[x] = RSIIndicator(df[x], window=2).rsi().iloc[-1]
            sma200[x] = SMAIndicator(df[x], window=200).sma_indicator().iloc[-1]

    w = [0]  # adding data for CASH first

    for x in columns:
        if x == 'CASH':
            continue
        elif rsi2[x] < 5 and df[x].iloc[-1] > sma200[x]:
            w.append(1)
        elif rsi2[x] > 95 and df[x].iloc[-1] < sma200[x]:
            w.append(-1)
        else:
            w.append(0)

    w = np.array(w)
    if np.sum(abs(w)) == 0:
        weights = w
        weights[0] = 1  # hold cash if all weights = 0
    else:
        weights = w / np.nansum(abs(w))

    return weights, settings


def bl_allocation(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets = CLOSE.shape[1]
    markets = settings["markets"]
    viewdict = {}
    filtered_closed = {}
    models = [f.split(".model")[0] for f in os.listdir("./data/xgb") if f.endswith(".model")]
    future_index = []
    for i in range(0, nMarkets):
        future_name = markets[i]
        if future_name in models:
            with open(f'./data/xgb/{future_name}.txt') as f:
                ifi = json.load(f)
            if np.count_nonzero(np.isnan(np.transpose(CLOSE)[i])) != 0:
                continue

            filtered_closed[future_name] = np.transpose(CLOSE)[i]
            future_index.append(i)
            df = pd.DataFrame(np.transpose(
                [
                    np.transpose(OPEN)[i],
                    np.transpose(HIGH)[i],
                    np.transpose(LOW)[i],
                    np.transpose(CLOSE)[i],
                    np.transpose(VOL)[i]
                ]),
                columns=["Open", "High", "Low", "Close", "Vol"]
            )
            df = transform_data(df, ifi)
            features = df.iloc[-1:][ifi].to_numpy()
            model_dir = f"./data/xgb/{future_name}.model"
            prediction = predict(model_dir, features)[0]
            closing = np.transpose(CLOSE)[i][-1]
            pct_change = (closing - prediction) / closing
            viewdict[future_name] = pct_change

    filtered_closed = pd.DataFrame.from_dict(filtered_closed)
    if filtered_closed.empty:
        if settings['history']:
            pos = settings['history'][-1]
        else:
            pos = np.array([1 if i == 'CASH' else 0 for i in settings['markets']])
    else:
        try:
            cov_matrix = risk_models.exp_cov(filtered_closed)
            bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)
            rets = bl.bl_returns()
            ef = EfficientFrontier(rets, cov_matrix)
            optimised_weights = ef.max_sharpe()
            weights = np.array([value for key, value in optimised_weights.items()])
            pos = np.zeros(nMarkets)
            for j in range(0, len(weights)):
                pos[future_index[j]] = weights[j]
            print(f"Positions {pos} at day {settings['day']}")
        except Exception:
            print(f"No optimal solution, using same weights allocation on day {settings['day']}")
            if settings['history']:
                pos = settings['history'][-1]
            else:
                pos = np.array([1 if i == 'CASH' else 0 for i in settings['markets']])
    settings['history'].append(pos)
    return pos, settings


def arima(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    # indicate here test or train data
    MODE = "TEST"
    df = pd.DataFrame(CLOSE)
    df.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)
    df = df[list(filter(lambda x: x != 'CASH', settings['markets']))]
    df = np.log(df)
    columns = list(df)
    mu = [0]
    # note that the fitted linear models for the following futures need to be trained when reproducing results
    # cannot save models to github due to large size
    # code to train and fit model can be found at ARIMA.py
    futures_List = ["F_ED", "F_ZQ", "F_TU", "F_SS", "F_EB"]
    if MODE == 'TRAIN':
        for i in columns:
            close = df[i].fillna("None").tolist()[-1]
            if i in futures_List:
                df_in_sample = pd.read_csv('ARIMA/In_sample_predictions/{}.csv'.format(i))
                try:
                    pred = df_in_sample[df_in_sample['date'] == DATE[-1]]['predictions'].tolist()[0]
                except:
                    pred = 0
            else:
                pred = 0
            mu.append(pred)
    else:
        test_index = np.where(DATE == 20201231)[0][0]
        for i in columns:
            if i in futures_List:
                # train and fit ARIMA models first
                with open('ARIMA/Models/{}.pkl'.format(i), 'rb') as pkl:
                    model = pickle.load(pkl)
                    test = df[i].loc[test_index:]
                    try:
                        model.update(test)
                    except:
                        print("cant update model")
                    try:
                        pred = model.predict(n_periods=1)[0]
                    except:
                        print("cant predict")
                        pred = 0
            else:
                pred = 0
            mu.append(pred)
    columns.insert(0, "CASH")
    mu = pd.Series(mu, index=columns)
    S = risk_models.sample_cov(df)
    S.insert(loc=0, column='CASH', value=0)
    cash = functools.reduce(lambda a, b: {**a, **b}, [{ticker: [0]} for ticker in settings['markets']])
    S = pd.DataFrame(cash, index=['CASH']).append(S)
    ef = EfficientFrontier(mu, S)

    try:
        optimised_weights = ef.max_sharpe()
        weights = np.array([value for key, value in optimised_weights.items()])
        print("Optimal solution found!")
    except Exception:
        print("No optimal solution, using same weights allocation in the previous timestep")
        if settings['history']:
            weights = settings['history'][-1]
        else:
            weights = np.array([1 if i == 'CASH' else 0 for i in settings['markets']])
    settings['history'].append(weights)
    return weights, settings