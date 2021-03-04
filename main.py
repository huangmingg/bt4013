import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import functools 

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets = CLOSE.shape[1]
    df = pd.DataFrame(CLOSE)
    df.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)

    daily_return = (np.log(df) - np.log(df.shift(1))).iloc[1:]
    daily_return.drop(['CASH'], axis=1, inplace=True)

    print(daily_return)

    # To replace with expected mean from optimized AR/MA parameters
    mu = expected_returns.mean_historical_return(daily_return)
    # Add cash back into the expected returns matrix at 0
    mu = pd.Series([0], index=['CASH']).append(mu)

    S = risk_models.sample_cov(daily_return)
    # Add cash back into the covariance matrix
    S.insert(loc=0, column='CASH', value=0)
    cash = functools.reduce(lambda a,b: {**a, **b }, [{ticker: [0]} for ticker in settings['markets']])
    S = pd.DataFrame(cash, index=['CASH']).append(S)
    ef = EfficientFrontier(mu, S)
    optimised_weights = ef.max_sharpe()
    weights = [value for key, value in optimised_weights.items()]
    return weights, settings
    
def mySettings():
    """ Define your trading system settings here """

    # Allowed futures
    # futuresList = ['F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT','F_DX','F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY','F_KC','F_LB','F_LC','F_LN','F_MD','F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL','F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD','F_F']

    settings = {
        'markets': ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC'], 'lookback': 30, 'budget': 10 ** 6, 'slippage': 0.05}
    settings['beginInSample'] = '20191201'
    settings['endInSample'] = '20201231'

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
