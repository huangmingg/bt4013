import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
# from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import functools


def simple_moving_average(daily_return):
    return daily_return[-50:].mean()*252


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets = CLOSE.shape[1]
    df = pd.DataFrame(CLOSE)
    df.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)

    # daily_return = df.pct_change().dropna(how="all")
    # daily_return.drop(['CASH'], axis=1, inplace=True)
    #
    # columns = list(daily_return)
    # mu = pd.Series()
    # mu = pd.Series([0], index=['CASH'])
    #
        # for i in columns:
        #     # fitting ARMA model first
        #     # assume stationarity, d=0, no seasonality
        #     res = simple_moving_average(daily_return[i])
        #     # print(res)
        #     # model = auto_arima(daily_return[i], start_p=1, start_q=1, max_p=5, max_q=5, d=0, seasonal=False, stationary=False,
        #     #                suppress_warnings=True, error_action='warn', stepwise=True)
        #     # pred = model.predict(n_periods=1)[0]
        #     expected = pd.Series([res], index=[i])
        #     mu = mu.append(expected)

    df = df[list(filter(lambda x: x != 'CASH', settings['markets']))]
    # mu = pd.Series([0], index=['CASH']).append(expected_returns.ema_historical_return(df, compounding=False))
    mu = pd.Series([0], index=['CASH']).append(expected_returns.mean_historical_return(df, compounding=False))
    mu.fillna(0, inplace=True)
    S = CovarianceShrinkage(df).ledoit_wolf()
    # Add cash back into the covariance matrix
    S.insert(loc=0, column='CASH', value=0)
    cash = functools.reduce(lambda a, b: {**a, **b}, [{ticker: [0]} for ticker in settings['markets']])
    S = pd.DataFrame(cash, index=['CASH']).append(S)
    ef = EfficientFrontier(mu, S)
    try:
        optimised_weights = ef.max_sharpe()
        weights = np.array([value for key, value in optimised_weights.items()])
        print("Optimal solution found!")
    except Exception as e:
        print("No optimal solution, using same weights allocation in the previous time step")
        print(settings['invalid_solution'])
        settings['invalid_solution'] += 1
        if settings['history']:
            weights = settings['history'][-1]
        else:
            weights = np.array([1 if i == 'CASH' else 0 for i in settings['markets']])
    # baseline
    # weights = np.array([0 if i == 'CASH' else (1/88) for i in settings['markets']])
    settings['history'].append(weights)
    return weights, settings


def mySettings():
    """ Define your trading system settings here """

    # Allowed futures
    # futuresList = ['F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT','F_DX','F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY','F_KC','F_LB','F_LC','F_LN','F_MD','F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL','F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD','F_F']

    settings = {
        'markets': ['CASH','F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT','F_DX','F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY','F_KC','F_LB','F_LC','F_LN','F_MD','F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL','F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD','F_F'],
        'lookback': 504, 'budget': 10 ** 6, 'slippage': 0.05,
        'beginInSample': '20190101', 'endInSample': '20210225', 'history': [], 'invalid_solution': 0}

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
