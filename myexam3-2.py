from pykrx import stock
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 삼성전자 주식 데이터 불러오기
start_date = "20230101"
end_date = "20240131"
ticker = "005930"  # 삼성전자 종목 코드

df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['날짜'])
df = df.set_index('Date')

# 특성 엔지니어링
df['전일종가'] = df['종가'].shift(1)
df['전일거래량'] = df['거래량'].shift(1)
df['다음날고가'] = df['고가'].shift(-1)

# NaN 값 제거
df = df.dropna()

# 학습 데이터와 테스트 데이터 분리
train_data = df[:'2024-01-15']
test_data = df['2024-01-16':]

# 특성과 타겟 분리
X_train = train_data[['시가', '종가', '저가', '고가', '거래량', '전일종가', '전일거래량']]
y_train = train_data['다음날고가']

X_test = test_data[['시가', '종가', '저가', '고가', '거래량', '전일종가', '전일거래량']]
y_test = test_data['다음날고가']

# 랜덤 포레스트 모델 학습
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = rf_model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['고가'], label='실제 고가')
plt.plot(test_data.index, y_pred, label='예측 고가', color='red')
plt.title('삼성전자 주식 고가 예측')
plt.xlabel('날짜')
plt.ylabel('가격')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()