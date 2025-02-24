import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 데이터 불러오기
def get_stock_data(ticker, start_date, end_date):
    df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
    return df

# 변동성 돌파 전략 구현
def volatility_breakout_strategy(df, k):
    df['Range'] = df['High'] - df['Low']
    df['Target'] = df['Open'] + df['Range'].shift(1) * k
    df['Position'] = 0
    df.loc[df['High'] > df['Target'], 'Position'] = 1
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
    return df

# 백테스팅 함수
def backtest(df, initial_capital):
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cumulative_Wealth'] = df['Cumulative_Returns'] * initial_capital
    return df

# 메인 실행 코드
if __name__ == "__main__":
    ticker = "005930"  # 삼성전자 종목 코드
    start_date = "20230101"
    end_date = "20231231"
    k = 0.5  # 변동성 계수
    initial_capital = 10000000  # 초기 자본금 (1천만원)

    # 데이터 불러오기
    data = get_stock_data(ticker, start_date, end_date)

    # 전략 적용 및 백테스팅
    result = volatility_breakout_strategy(data, k)
    backtest_result = backtest(result, initial_capital)

    # 결과 출력
    print("백테스팅 결과:")
    print(f"총 수익률: {(backtest_result['Cumulative_Returns'].iloc[-1] - 1) * 100:.2f}%")
    print(f"최종 자산: {backtest_result['Cumulative_Wealth'].iloc[-1]:,.0f}원")

    # 1. 누적 수익률 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_result.index, backtest_result['Cumulative_Wealth'])
    plt.title('변동성 돌파 전략 백테스팅 결과')
    plt.xlabel('날짜')
    plt.ylabel('자산')
    plt.grid(True)
    plt.show()

    # 2. 일별 수익률 시각화
    buy_signals = result[result['Position']==1] # 매수 신호가 있는 날짜만 필터링
    plt.figure(figsize=(14, 7))
    plt.bar(buy_signals.index, buy_signals['Returns'] * 100, color='green')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.title('일별 수익률')
    plt.xlabel('날짜')
    plt.ylabel('일별 수익률 (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()