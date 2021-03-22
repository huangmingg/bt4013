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

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    settings['day'] += 1
    nMarkets = CLOSE.shape[1]
    markets = settings["markets"]

    if settings['strategy'] == "baseline":
        pos = np.array([1 / 89 for i in settings['markets']])
        return pos, settings

    elif settings['strategy'] == 'sma':
        nMarkets = CLOSE.shape[1]

        periodLonger = 200
        periodShorter = 40

        # Calculate Simple Moving Average (SMA)
        smaLongerPeriod = np.nansum(CLOSE[-periodLonger:, :], axis=0) / periodLonger
        smaShorterPeriod = np.nansum(CLOSE[-periodShorter:, :], axis=0) / periodShorter

        longEquity = smaShorterPeriod > smaLongerPeriod
        shortEquity = ~longEquity

        pos = np.zeros(nMarkets)
        pos[longEquity] = 1
        pos[shortEquity] = -1

        weights = pos / np.nansum(abs(pos))
        print(weights)

        return weights, settings

    elif settings['strategy'] == "bl_allocation":
        viewdict = {}
        filtered_closed = {}
        models = [f.split(".model")[0] for f in os.listdir("./data/xgb") if f.endswith(".model")]
        future_index = []
        for i in range(0, nMarkets - 1):
            future_name = markets[i + 1]
            if future_name in models:
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
                df = transform_data(df)
                features = df.iloc[-1:][REQUIRED_COLS].to_numpy()
                model_dir = f"./data/xgb/{markets[i + 1]}.model"
                prediction = predict(model_dir, features)[0]
                closing = np.transpose(CLOSE)[i][-1]
                pct_change = (closing - prediction) / closing
                viewdict[future_name] = pct_change

        filtered_closed = pd.DataFrame.from_dict(filtered_closed).iloc[:30]
        cov_matrix = risk_models.exp_cov(filtered_closed)
        bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)
        rets = bl.bl_returns()
        ef = EfficientFrontier(rets, cov_matrix)

        try:
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

    elif settings['strategy'] == "arima":
        df = pd.DataFrame(CLOSE)
        df.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)
        daily_return = (np.log(df) - np.log(df.shift(1))).iloc[1:]
        # daily_return.drop(['CASH'], axis=1, inplace=True)

        columns = list(daily_return)
        # mu = pd.Series()
        mu = pd.Series([0], index=['CASH'])

        # for i in columns:
        #     # fitting ARMA model first
        #     # assume stationarity, d=0, no seasonality
        #     model = auto_arima(daily_return[i], start_p=0, start_q=0, max_p=5, max_q=5, d=0, seasonal=False,
        #                        stationary=True, suppress_warnings=True, error_action='warn', stepwise=True)
        #     pred = model.predict(n_periods=1)[0]
        #     expected = pd.Series([pred], index=[i])
        #     mu = mu.append(expected)

        mu = mu.append(expected_returns.ema_historical_return(df))
        df = df[list(filter(lambda x: x != 'CASH', settings['markets']))]
        # S = CovarianceShrinkage(df).ledoit_wolf()
        S = risk_models.sample_cov(df)
        print(S)
        # Add cash back into the covariance matrix
        S.insert(loc=0, column='CASH', value=0)
        print(S)
        cash = functools.reduce(lambda a, b: {**a, **b}, [{ticker: [0]} for ticker in settings['markets']])
        S = pd.DataFrame(cash, index=['CASH']).append(S)
        print(S)
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


def mySettings():
    futures_list = ['F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD', 'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC', 'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP', 'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU', 'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_DT', 'F_UB', 'F_UZ', 'F_GS', 'F_LX', 'F_SS', 'F_DL', 'F_ZQ', 'F_VX', 'F_AE', 'F_BG', 'F_BC', 'F_LU', 'F_DM', 'F_AH', 'F_CF', 'F_DZ', 'F_FB', 'F_FL', 'F_FM', 'F_FP', 'F_FY', 'F_GX', 'F_HP', 'F_LR', 'F_LQ', 'F_ND', 'F_NY', 'F_PQ', 'F_RR', 'F_RF', 'F_RP', 'F_RY', 'F_SH', 'F_SX', 'F_TR', 'F_EB', 'F_VF', 'F_VT', 'F_VW', 'F_GD', 'F_F']

    # possible strategies - add on here
    # STRATEGIES = ['baseline', 'bl_allocation', 'arima', 'sma']
    settings = {'markets': futures_list,
                'lookback': 504,
                'budget': 10 ** 6,
                'slippage': 0.05,
                'beginInSample': '19900101',
                'endInSample': '20210225',
                'day': 0,
                'history': [],
                'strategy': 'sma',
                }

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
