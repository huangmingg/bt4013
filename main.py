import numpy
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):

    nMarkets = CLOSE.shape[1]
    # print(CLOSE)
    # print(exposure)
    # print(equity)

    periodLonger = 200
    periodShorter = 40

    # Calculate Simple Moving Average (SMA)
    smaLongerPeriod = numpy.nansum(CLOSE[-periodLonger:, :], axis=0) / periodLonger
    smaShorterPeriod = numpy.nansum(CLOSE[-periodShorter:, :], axis=0) / periodShorter

    longEquity = smaShorterPeriod > smaLongerPeriod
    shortEquity = ~longEquity

    pos = numpy.zeros(nMarkets)
    pos[longEquity] = 1
    pos[shortEquity] = -1

    ## Calculate the expected return and covariance matrix for the next time step.
    ## Parse in the values into a sharpe ratio optimizer to get the weighted matrix that will be returned from the myTradingSystem function.


    daily_return = pd.DataFrame(CLOSE)
    daily_return.rename(lambda x: settings['markets'][x], axis='columns', inplace=True)
    daily_return.drop(['CASH'], axis=1, inplace=True)
    print(daily_return)
    # res = pd.DataFrame(CLOSE)
    # print(daily_return)
    # res.fillna(0, inplace=True)
    mu = expected_returns.mean_historical_return(daily_return)
    S = risk_models.sample_cov(daily_return)
    # print(S)

    ef = EfficientFrontier(mu, S)
    test_weights = ef.max_sharpe()
    # print(test_weights)
    # ef.portfolio_performance(verbose=True)


    weights = pos / numpy.nansum(abs(pos))
    print(weights, settings)
    return weights, settings


def mySettings():
    """ Define your trading system settings here """

    # Allowed futures
    # futuresList = ['F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT','F_DX','F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY','F_KC','F_LB','F_LC','F_LN','F_MD','F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL','F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD','F_F']

    settings = {
        'markets': ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC'], 'lookback': 5, 'budget': 10 ** 6, 'slippage': 0.05}
    settings['beginInSample'] = '20201201'
    settings['endInSample'] = '20201231'

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
