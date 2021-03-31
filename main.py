from strategies import *


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    settings['day'] += 1
    print(DATE[-1])
    nMarkets = CLOSE.shape[1]
    markets = settings["markets"]

    if settings['strategy'] == "baseline":
        return baseline(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] == 'sma':
        return sma_504(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] == "ema":
        return ema_504(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] == 'trend':
        return trend_following(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] == 'pairs_trade':
        return pairs_trade(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] == "obv":
        return on_balance_volume(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] == "mean_rev":
        return mean_reversion(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] == "bl_allocation":
        return bl_allocation(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] == "arima":
        return arima(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] == 'ensem':
        weights1, _ = trend_following(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)
        weights2, _ = on_balance_volume(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)
        weights3, _ = sma_504(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)
        weights4, _ = baseline(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)
        totalw = weights1 + weights2 + weights3 + weights4
        totalw = totalw/np.nansum(abs(totalw))
        return totalw, _




def mySettings():
    futures_list = ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD', 'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC', 'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP', 'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU', 'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_DT', 'F_UB', 'F_UZ', 'F_GS', 'F_LX', 'F_SS', 'F_DL', 'F_ZQ', 'F_VX', 'F_AE', 'F_BG', 'F_BC', 'F_LU', 'F_DM', 'F_AH', 'F_CF', 'F_DZ', 'F_FB', 'F_FL', 'F_FM', 'F_FP', 'F_FY', 'F_GX', 'F_HP', 'F_LR', 'F_LQ', 'F_ND', 'F_NY', 'F_PQ', 'F_RR', 'F_RF', 'F_RP', 'F_RY', 'F_SH', 'F_SX', 'F_TR', 'F_EB', 'F_VF', 'F_VT', 'F_VW', 'F_GD', 'F_F']

    # possible strategies - add on here
    # STRATEGIES = ['baseline', 'bl_allocation', 'arima', 'sma', 'ema', 'pairs_trade', 'obv', 'mean_rev', 'trend', 'ensem]
    # MODE = "TEST" / "TRAIN"
    MODE = "TEST"


    train_date = {
        'beginInSample': '19900101',
        'endInSample': '20201231',
    }

    test_date = {
        'beginInSample': '20190123',
        'endInSample': '20210303',
    }

    dates = train_date if MODE == "TRAIN" else test_date

    settings = {'markets': futures_list,
                'lookback': 504,
                'budget': 10 ** 6,
                'slippage': 0.05,
                **dates,
                'day': 0,
                'history': [],
                'strategy': 'ensem',
                }

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
