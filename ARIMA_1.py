import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima.arima import ndiffs
import pmdarima as pm

from datetime import datetime, timedelta

def forecast_one_step():
    fc, conf_int = model.predict(n_periods=1 # 한 스텝씩!
        , return_conf_int=True)              # 신뢰구간 출력
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0]
    )

dayinit = datetime.today()

forecasts = []
y_pred = []
pred_upper = []
pred_lower = []

today = dayinit.strftime("%Y-%m-%d")
yesterday = (dayinit - timedelta(365)).strftime("%Y-%m-%d")
nextday = (dayinit + timedelta(90)).strftime("%Y-%m-%d")

df_krx = fdr.StockListing('KRX') # 한국거래소 상장종목 전체
ticker = df_krx[df_krx.Name=='삼성전자']['Symbol'] # 티커
ss = fdr.DataReader(''.join(ticker.values), yesterday, today)
ss1 =fdr.DataReader(''.join(ticker.values), today, nextday)

y_train = ss['Close'][:int(0.7*len(ss))]
y_test = ss['Close'][int(0.7*len(ss)):]
y_next.append()


model = pm.auto_arima(y_train, d = 1, seasonal = False, trace = True)
model.fit(y_train)

for i in range(1, 60):
    y_next.append([(dayinit + timedelta(i)).strftime("%Y-%m-%d"), 0])
    


print(y_test)

#

for new_ob in y_next:
    fc, conf = forecast_one_step()
    y_pred.append(fc)
    pred_upper.append(conf[1])
    pred_lower.append(conf[0])

    ## 모형 업데이트 !!
    model.update(new_ob)

y_pred = pd.DataFrame({"test": y_test, "pred": y_pred})

fig, axes = plt.subplots(1,1,figsize = (12, 4))

plt.plot(y_train, label = 'Train')
plt.plot(y_test, label = 'Test')
plt.plot(y_pred, label = 'pred')
plt.legend()

#
#print(model.summary())
#model.plot_diagnostics(figsize=(16, 8))


plt.show()


