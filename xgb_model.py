import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from ta import add_all_ta_features
import math


REQUIRED_COLS = ['momentum_rsi', 'momentum_stoch', 'trend_cci', 'trend_ema_fast', 'trend_ema_slow',
                'volatility_bbh', 'volatility_bbl']


def transform_data(df):
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Vol", fillna=True)
    df['Target'] = df['Close'].shift(-1)
    data = df[REQUIRED_COLS + ['Target']]
    return data


def get_rmse(a, b):
    return math.sqrt(np.mean((np.array(a)-np.array(b))**2))


def format_header(df):
    df.columns = [s.title().strip() for s in df.columns]
    return df


def train_xgb_regressor(future_list):
    model_store = []
    for future in future_list:
        df = pd.read_csv(f"./tickerData/{future}.txt")
        df = format_header(df)
        df = df.loc[df['Date'] < 20210101]
        data = transform_data(df)
        data = data.iloc[:-1]

        train_df, test_df = train_test_split(data, test_size=0.2, shuffle=False)
        train_x = train_df[REQUIRED_COLS].to_numpy()
        train_y = train_df['Target'].to_numpy()
        test_x = test_df[REQUIRED_COLS].to_numpy()
        test_y = test_df['Target'].to_numpy()
        model = XGBRegressor(
            objective='reg:squarederror',
            seed=69,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=1,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            gamma=0
        )

        ## Tune the model

        # Train the model
        model.fit(train_x, train_y)
        est = model.predict(test_x)
        rmse = get_rmse(test_y, est)
        model_store.append((model, rmse, future))

    model_store = sorted(model_store, key=lambda x: x[1], reverse=True)[:20]
    for i in model_store:
        model.save_model(f'./data/xgb/{i[2]}.model')


def predict(model_dir, features):
    model = XGBRegressor()
    model.load_model(model_dir)
    return model.predict(features)


if __name__ == '__main__':
    futures = ['F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD', 'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC', 'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP', 'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU', 'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_DT', 'F_UB', 'F_UZ', 'F_GS', 'F_LX', 'F_SS', 'F_DL', 'F_ZQ', 'F_VX', 'F_AE', 'F_BG', 'F_BC', 'F_LU', 'F_DM', 'F_AH', 'F_CF', 'F_DZ', 'F_FB', 'F_FL', 'F_FM', 'F_FP', 'F_FY', 'F_GX', 'F_HP', 'F_LR', 'F_LQ', 'F_ND', 'F_NY', 'F_PQ', 'F_RR', 'F_RF', 'F_RP', 'F_RY', 'F_SH', 'F_SX', 'F_TR', 'F_EB', 'F_VF', 'F_VT', 'F_VW', 'F_GD', 'F_F']
    train_xgb_regressor(futures)
