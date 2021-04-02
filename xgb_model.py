import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from ta import add_all_ta_features
import math
import json


REQUIRED_COLS = ['volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_mfi',
       'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_nvi', 'volume_vwap',
       'volatility_atr', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
       'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
       'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
       'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
       'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
       'volatility_dcw', 'volatility_dcp', 'volatility_ui', 'trend_macd',
       'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
       'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_adx',
       'trend_adx_pos', 'trend_adx_neg', 'trend_vortex_ind_pos',
       'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix',
       'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',
       'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
       'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
       'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
       'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
       'trend_psar_down', 'trend_psar_up_indicator',
       'trend_psar_down_indicator', 'trend_stc', 'momentum_rsi',
       'momentum_stoch_rsi', 'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d',
       'momentum_tsi', 'momentum_uo', 'momentum_stoch',
       'momentum_stoch_signal', 'momentum_wr', 'momentum_ao', 'momentum_kama',
       'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal', 'momentum_ppo_hist']


def transform_data(df, required_cols):
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Vol", fillna=True)
    df['Target'] = df['Close'].shift(-1)
    data = df[required_cols + ['Target']]
    data = data.iloc[:-1]
    return data


def get_rmse(a, b):
    return math.sqrt(np.mean((np.array(a)-np.array(b))**2))


def format_header(df):
    df.columns = [s.title().strip() for s in df.columns]
    return df


def train_model(data, required_cols):
    train_df, test_df = train_test_split(data, test_size=0.2, shuffle=False)
    train_x = train_df[required_cols].to_numpy()
    train_y = train_df['Target'].to_numpy()
    test_x = test_df[required_cols].to_numpy()
    test_y = test_df['Target'].to_numpy()
    model = XGBRegressor(
        objective='reg:squarederror',
        seed=1,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        gamma=0
    )

    model.fit(train_x, train_y)
    est = model.predict(test_x)
    rmse = get_rmse(test_y, est)
    rmse = rmse / data['Target'].mean()
    return model, rmse


def train_xgb_regressor(future_list):
    model_store = []
    for future in future_list:
        df = pd.read_csv(f"./tickerData/{future}.txt")
        df = format_header(df)
        df = df.loc[df['Date'] < 20210101]
        data = transform_data(df, REQUIRED_COLS)
        # Obtain naive model with all TA indicators
        model, _ = train_model(data, REQUIRED_COLS)
        # Keep only the important features
        fi = sorted([(REQUIRED_COLS[index], score) for index, score in enumerate(model.feature_importances_)], key=lambda x: x[1], reverse=True)
        ifi = list(map(lambda x: x[0], filter(lambda x: x[1] > 0.1, fi)))
        new_data = transform_data(df, ifi)
        new_model, rmse = train_model(new_data, ifi)
        model_store.append((new_model, rmse, future, ifi))

    model_store = sorted(model_store, key=lambda x: x[1], reverse=False)[:20]
    for i in model_store:
        i[0].save_model(f'./data/xgb/{i[2]}.model')
        with open(f'./data/xgb/{i[2]}.txt', 'w') as file:
            json.dump(i[3], file)


def predict(model_dir, features):
    model = XGBRegressor()
    model.load_model(model_dir)
    return model.predict(features)


if __name__ == '__main__':
    futures = ['F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD', 'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC', 'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP', 'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU', 'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_DT', 'F_UB', 'F_UZ', 'F_GS', 'F_LX', 'F_SS', 'F_DL', 'F_ZQ', 'F_VX', 'F_AE', 'F_BG', 'F_BC', 'F_LU', 'F_DM', 'F_AH', 'F_CF', 'F_DZ', 'F_FB', 'F_FL', 'F_FM', 'F_FP', 'F_FY', 'F_GX', 'F_HP', 'F_LR', 'F_LQ', 'F_ND', 'F_NY', 'F_PQ', 'F_RR', 'F_RF', 'F_RP', 'F_RY', 'F_SH', 'F_SX', 'F_TR', 'F_EB', 'F_VF', 'F_VT', 'F_VW', 'F_GD', 'F_F']
    train_xgb_regressor(futures)
