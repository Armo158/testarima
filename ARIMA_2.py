import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima.arima import ndiffs
import pmdarima as pm

from datetime import datetime, timedelta

dayinit = datetime.today()

today = dayinit.strftime("%Y-%m-%d")
yesterday = (dayinit - timedelta(365)).strftime("%Y-%m-%d")
#nextday = (dayinit + timedelta(60)).strftime("%Y-%m-%d")

df_krx = fdr.StockListing('KRX') # 한국거래소 상장종목 전체
ticker = df_krx[df_krx.Name=='HLB']['Symbol'] # 티커
ss = fdr.DataReader(''.join(ticker.values), yesterday, today)

y_train = ss['Close'][:int(0.7*len(ss))]
y_test = ss['Close'][int(0.7*len(ss)):]


model = pm.auto_arima(y_train, d = 1, seasonal = False, trace = True)
model.fit(y_train)


y_predict = model.predict(n_periods = len(y_test))
y_predict = pd.DataFrame(y_predict,index = y_test.index,columns=['Prediction'])


fig, axes = plt.subplots(1,1,figsize = (12, 4))

plt.plot(y_train, label = 'Train')
plt.plot(y_test, label = 'Test')
plt.plot(y_predict, label = 'Prediction')
plt.legend()

#print(model.summary())
#model.plot_diagnostics(figsize=(16, 8))


plt.show()
