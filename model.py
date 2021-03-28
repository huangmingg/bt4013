import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from pmdarima.arima import auto_arima
import pickle
import joblib
import numpy as np
from pypfopt import expected_returns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os

futures_List = ['F_NR','F_O','F_OJ','F_PA','F_PL','F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD','F_F']
futures_List_reduced = ["F_EB", "F_ED", "F_F", "F_SS", "F_VW", "F_ZQ"]
for i in futures_List:
	os.chdir('/Users/chengjunlam/Desktop/Github local repos/bt4013')
	df = pd.read_csv("tickerData/{}.txt".format(i))
	df['DATE']= pd.to_datetime(df['DATE'],format='%Y%m%d')
	df = df.loc[(df['DATE'] > '19900101') & (df['DATE'] <= '20201231')]	
	try:
		df = df[' CLOSE']
	except:
		df = df['CLOSE']
	daily_return = np.log(df)
	model = auto_arima(daily_return, start_p=0, start_q=0, max_p=5, max_q=5,d = 1, max_d = 5,start_P = 0,D=1,start_Q=0,max_P=5,max_Q=5,max_D=5,m=5, seasonal=True, trace=True,suppress_warnings=True, error_action='warn', stepwise=True,n_fits=50)
	print("+++++++++++++++++++++++")
	print(model.summary())
	# save model
	os.chdir('/Volumes/Seagate Backup Plus Drive/BT4013 /')
	with open('{}.pkl'.format(i), 'wb') as pkl:
		pickle.dump(model, pkl)
# import os
# os.chdir('/Volumes/Seagate Backup Plus Drive/BT4013 /')
# with open('{}.pkl'.format("F_CC"), 'rb') as pkl:
#     model= pickle.load(pkl)
#     pred = model.predict(n_periods=1)[0]
#     print(pred)


