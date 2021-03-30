from pmdarima.arima import auto_arima
import pandas as pd
import pickle
import joblib
import numpy as np
import os

#selection and fitting of ARIMA/SARIMA models on the selected futures based on XGBR regression analysis
futures_List = ["F_ED", "F_ZQ", "F_TU", "F_SS", "F_EB"] 
for i in futures_List:
	df = pd.read_csv("tickerData/{}.txt".format(i))
	df['DATE']= pd.to_datetime(df['DATE'],format='%Y%m%d')
	#filter data to use only training data to fit model
	df = df.loc[(df['DATE'] > '19900101') & (df['DATE'] <= '20201231')]	
	try:
		df = df[' CLOSE']
	except:
		df = df['CLOSE']
	daily_return = np.log(df)
	#set period as 5 to represent 5 working days per period
	#set a range of 0 to 5 for all paramaters
	model = auto_arima(daily_return, start_p=0, start_q=0, max_p=5, max_q=5,d = 1, max_d = 5,start_P = 0,D=1,start_Q=0,max_P=5,max_Q=5,max_D=5,m=5, seasonal=True, trace=True,suppress_warnings=True, error_action='warn', stepwise=True,n_fits=50)
	# save model
	with open('ARIMA/Models/{}.pkl'.format(i), 'wb') as pkl:
		pickle.dump(model, pkl)

#saving in-sample prediction for evaluation of fitted model on training data
for i in futures_List:
	df = pd.read_csv("tickerData/{}.txt".format(i))
	df['DATE']= pd.to_datetime(df['DATE'],format='%Y%m%d')
	#filter data to use only training data to fit model
	df = df.loc[(df['DATE'] > '19900101') & (df['DATE'] <= '20201231')]
	pred = None
	with open('ARIMA/Models/{}.pkl'.format(i), 'rb') as pkl:
		model = pickle.load(pkl)
		pred = model.predict_in_sample(1)
	date = df["DATE"].tolist()
	date = [int(x.strftime("%Y%m%d")) for x in date]
	df_in_sample = pd.DataFrame({"predictions":pred, "date": date})
	df_in_sample.to_csv('ARIMA/In_sample_predictions/{}.csv'.format(i), index = False)









