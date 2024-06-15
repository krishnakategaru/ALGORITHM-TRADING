#import libraries
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
import warnings
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

warnings.filterwarnings("ignore")


# symbol_to_fetch = 'AAPL'
# start_date = '2020-01-01'
# end_date = '2024-05-01'
# # Parameters
# batch_size = 256
# sequence_length = 30

def fetch_ticker_data(symbol, start_date, end_date):
    """Fetches stock data for a given symbol using yfinance."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(start='1980-01-01', end=end_date)
    return data

def label_data(data):
    # Calculate the percentage change in price from one day to the next
    data['pr_change_on_last_day'] = data['Close'].pct_change()
    data['pr_change_on_current_day'] = data['pr_change_on_last_day'].shift(-1)
    data.iloc[0,-2] = 0
    data['sentiment'] = pd.Series(np.where(data['pr_change_on_current_day'] > 0, 1, 0), index=data.index)
    # data['perc_change'] = data['Percentage Change']
    # # Drop any rows with missing values
    # data.dropna(inplace=True)
    data.drop('pr_change_on_current_day',axis=1 , inplace=True)
    return data
def train_model(symbol_to_fetch,start_date,end_date,batch_size,sequence_length,stride =1):
    stock = fetch_ticker_data(symbol_to_fetch, start_date, end_date)

    stock['fast_ma'] = stock['Close'].rolling(window=20).mean()
    stock['slow_ma'] = stock['Close'].rolling(window=50).mean()
    stock['ma_signal'] = np.where(stock['fast_ma'] > stock['slow_ma'], 1, -1)

    # 2. Bollinger Bands
    stock['bollinger_high'] = stock['Close'].rolling(window=20).mean() + (2 * stock['Close'].rolling(window=20).std())
    stock['bollinger_low'] = stock['Close'].rolling(window=20).mean() - (2 * stock['Close'].rolling(window=20).std())
    stock['bollinger_signal'] = np.where(stock['Close'] <= stock['bollinger_low'], 1, np.where(stock['Close'] >= stock['bollinger_high'], -1, 0))

    # 3. Exponential Moving Average (EMA)
    stock['ema'] = stock['Close'].ewm(span=20, adjust=False).mean()
    stock['ema_signal'] = np.where(stock['Close'] > stock['ema'], 1, -1)

    # 4. Envelopes
    stock['envelope_high'] = stock['Close'].rolling(window=20).mean() * (1 + 0.05)
    stock['envelope_low'] = stock['Close'].rolling(window=20).mean() * (1 - 0.05)
    stock['envelope_signal'] = np.where(stock['Close'] <= stock['envelope_low'], 1, np.where(stock['Close'] >= stock['envelope_high'], -1, 0))

    # 5. MACD
    stock['macd_line'] = stock['Close'].ewm(span=12, adjust=False).mean() - stock['Close'].ewm(span=26, adjust=False).mean()
    stock['macd_signal_line'] = stock['macd_line'].ewm(span=9, adjust=False).mean()
    stock['macd_signal'] = np.where(stock['macd_line'] > stock['macd_signal_line'], 1, -1)

    # 6. RSI
    def calculate_rsi(data, rsi_period=14):
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    stock['rsi'] = calculate_rsi(stock)
    stock['rsi_signal'] = np.where(stock['rsi'] < 30, 1, np.where(stock['rsi'] > 70, -1, 0))

    # 7. ATR
    high_low = stock['High'] - stock['Low']
    high_close = np.abs(stock['High'] - stock['Close'].shift())
    low_close = np.abs(stock['Low'] - stock['Close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    stock['atr'] = tr.rolling(window=14).mean()

    # 8. Stochastic Oscillator
    low_14 = stock['Low'].rolling(window=14).min()
    high_14 = stock['High'].rolling(window=14).max()
    stock['stochastic'] = 100 * (stock['Close'] - low_14) / (high_14 - low_14)
    stock['stochastic_signal'] = np.where(stock['stochastic'] < 20, 1, np.where(stock['stochastic'] > 80, -1, 0))

    # 9. CCI
    tp = (stock['High'] + stock['Low'] + stock['Close']) / 3
    ma_tp = tp.rolling(window=20).mean()
    md = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    stock['cci'] = (tp - ma_tp) / (0.015 * md)
    stock['cci_signal'] = np.where(stock['cci'] < -100, 1, np.where(stock['cci'] > 100, -1, 0))

    # 10. Chaikin Oscillator
    stock['ad_line'] = (2 * stock['Close'] - stock['Low'] - stock['High']) / (stock['High'] - stock['Low']) * stock['Volume']
    stock['chaikin'] = stock['ad_line'].ewm(span=3, adjust=False).mean() - stock['ad_line'].ewm(span=10, adjust=False).mean()
    stock['chaikin_signal'] = np.where(stock['chaikin'] > 0, 1, -1)

    # 11. Williams %R
    stock['williams_r'] = (high_14 - stock['Close']) / (high_14 - low_14) * -100
    stock['williams_r_signal'] = np.where(stock['williams_r'] < -80, 1, np.where(stock['williams_r'] > -20, -1, 0))

    # 12. Momentum Indicator
    stock['momentum'] = stock['Close'] - stock['Close'].shift(14)
    stock['momentum_signal'] = np.where(stock['momentum'] > 0, 1, -1)

    # 13. Rate of Change (ROC)
    stock['roc'] = (stock['Close'] - stock['Close'].shift(14)) / stock['Close'].shift(14) * 100
    stock['roc_signal'] = np.where(stock['roc'] > 0, 1, -1)

    # 14. On-Balance Volume (OBV)
    stock['obv'] = (np.sign(stock['Close'].diff()) * stock['Volume']).cumsum()
    stock['obv_signal'] = np.where(stock['obv'] > stock['obv'].shift(), 1, -1)

    # 15. Accumulation/Distribution Line (ADL)
    adl = ((stock['Close'] - stock['Low']) - (stock['High'] - stock['Close'])) / (stock['High'] - stock['Low']) * stock['Volume']
    stock['adl'] = adl.cumsum()
    stock['adl_signal'] = np.where(stock['adl'] > stock['adl'].shift(), 1, -1)

    # 16. Parabolic SAR
    stock['sar'] = stock['Close'].shift() * 0.02 + stock['Close'] * (1 - 0.02)
    stock['sar_signal'] = np.where(stock['Close'] > stock['sar'], 1, -1)

    # 17. Keltner Channel
    stock['kc_middle'] = stock['Close'].rolling(window=20).mean()
    stock['kc_high'] = stock['kc_middle'] + 2 * stock['atr']
    stock['kc_low'] = stock['kc_middle'] - 2 * stock['atr']
    stock['kc_signal'] = np.where(stock['Close'] <= stock['kc_low'], 1, np.where(stock['Close'] >= stock['kc_high'], -1, 0))

    # 18. Donchian Channel
    stock['donchian_high'] = stock['High'].rolling(window=20).max()
    stock['donchian_low'] = stock['Low'].rolling(window=20).min()
    stock['donchian_signal'] = np.where(stock['Close'] > stock['donchian_high'], 1, np.where(stock['Close'] < stock['donchian_low'], -1, 0))

    # 19. Moving Average Envelope
    stock['ma_envelope_high'] = stock['Close'].rolling(window=20).mean() * 1.025
    stock['ma_envelope_low'] = stock['Close'].rolling(window=20).mean() * 0.975
    stock['ma_envelope_signal'] = np.where(stock['Close'] <= stock['ma_envelope_low'], 1, np.where(stock['Close'] >= stock['ma_envelope_high'], -1, 0))

    # 20. Hull Moving Average (HMA)
    def hma(data, window):
        half_length = int(window / 2)
        sqrt_length = int(np.sqrt(window))
        wma_half = data.rolling(half_length).mean()
        wma_full = data.rolling(window).mean()
        diff = 2 * wma_half - wma_full
        return diff.rolling(sqrt_length).mean()

    stock['hma'] = hma(stock['Close'], 20)
    stock['hma_signal'] = np.where(stock['Close'] > stock['hma'], 1, -1)

    # 21. Weighted Moving Average (WMA)
    weights = np.arange(1, 21)
    stock['wma'] = stock['Close'].rolling(window=20).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    stock['wma_signal'] = np.where(stock['Close'] > stock['wma'], 1, -1)

    # 22. Volume Weighted Average Price (VWAP)
    stock['vwap'] = (stock['Close'] * stock['Volume']).cumsum() / stock['Volume'].cumsum()
    stock['vwap_signal'] = np.where(stock['Close'] > stock['vwap'], 1, -1)

    # 23. Ulcer Index (UI)
    stock['ulcer'] = np.sqrt(((stock['Close'] - stock['Close'].rolling(window=14).max()) ** 2).rolling(window=14).mean())
    stock['ulcer_signal'] = np.where(stock['ulcer'] < stock['ulcer'].shift(), 1, -1)

    # 24. Chande Momentum Oscillator (CMO)
    delta = stock['Close'].diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    stock['cmo'] = 100 * (up.rolling(window=14).sum() - down.rolling(window=14).sum()) / (up.rolling(window=14).sum() + down.rolling(window=14).sum())
    stock['cmo_signal'] = np.where(stock['cmo'] > 50, -1, np.where(stock['cmo'] < -50, 1, 0))

    # # 25. Fisher Transform
    # stock['median'] = (stock['High'] + stock['Low']) / 2
    # stock['fisher'] = np.arctanh(2 * (stock['median'] - stock['median'].rolling(window=20).min()) / (stock['median'].rolling(window=20).max() - stock['median'].rolling(window=20).min()) - 1)
    # stock['fisher_signal'] = np.where(stock['fisher'] > 0, 1, -1)

    # # 26. Market Momentum
    # market['momentum'] = market['Close'].rolling(window=200).mean()
    # stock['market_momentum'] = market['momentum']
    # stock['market_momentum_signal'] = np.where(stock['market_momentum'] > stock['market_momentum'].shift(), 1, -1)

    # 27. Heikin-Ashi
    stock['ha_close'] = (stock['Open'] + stock['High'] + stock['Low'] + stock['Close']) / 4
    stock['ha_open'] = (stock['Open'].shift() + stock['Close'].shift()) / 2
    stock['ha_high'] = stock[['High', 'ha_open', 'ha_close']].max(axis=1)
    stock['ha_low'] = stock[['Low', 'ha_open', 'ha_close']].min(axis=1)
    stock['ha_signal'] = np.where(stock['ha_close'] > stock['ha_open'], 1, -1)

    # 28. Ichimoku Cloud
    stock['tenkan_sen'] = (stock['High'].rolling(window=9).max() + stock['Low'].rolling(window=9).min()) / 2
    stock['kijun_sen'] = (stock['High'].rolling(window=26).max() + stock['Low'].rolling(window=26).min()) / 2
    stock['senkou_span_a'] = ((stock['tenkan_sen'] + stock['kijun_sen']) / 2).shift(26)
    stock['senkou_span_b'] = ((stock['High'].rolling(window=52).max() + stock['Low'].rolling(window=52).min()) / 2).shift(26)
    stock['chikou_span'] = stock['Close'].shift(-26)
    stock['ichimoku_signal'] = np.where(stock['Close'] > stock['senkou_span_a'], 1, -1)

    # 29. TRIX
    stock['trix'] = stock['Close'].ewm(span=15, adjust=False).mean()
    stock['trix'] = stock['trix'].ewm(span=15, adjust=False).mean()
    stock['trix'] = stock['trix'].ewm(span=15, adjust=False).mean()
    stock['trix_signal'] = np.where(stock['trix'] > stock['trix'].shift(), 1, -1)

    # 30. Price Rate of Change (PROC)
    stock['proc'] = (stock['Close'] - stock['Close'].shift(12)) / stock['Close'].shift(12)
    stock['proc_signal'] = np.where(stock['proc'] > 0, 1, -1)

    # Combined Signal Strategies
    # Strategy 1: Trend Confirmation with Momentum
    stock['combined_signal_1'] = np.where(
        (stock['ma_signal'] == 1) & (stock['macd_signal'] == 1) & (stock['rsi'] > 50) & (stock['rsi'] < 70), 
        1, 
        np.where((stock['ma_signal'] == -1) & (stock['macd_signal'] == -1) & (stock['rsi'] < 50) & (stock['rsi'] > 30), -1, 0)
    )

    # Strategy 2: Volatility Breakout with Trend
    stock['combined_signal_2'] = np.where(
        (stock['bollinger_signal'] == 1) & (stock['atr'] > stock['atr'].rolling(window=14).mean()) & (stock['sar_signal'] == 1), 
        1, 
        np.where((stock['bollinger_signal'] == -1) & (stock['atr'] > stock['atr'].rolling(window=14).mean()) & (stock['sar_signal'] == -1), -1, 0)
    )
    # 

    stock = stock.fillna(method="ffill", axis=0)
    stock = stock.fillna(method="bfill", axis=0)
    # stock.index = stock.index.date
    # Add date-related features
    stock['Year'] = stock.index.year
    stock['Month'] = stock.index.month
    stock['Day'] = stock.index.day
    stock['Weekday'] = stock.index.weekday

    # Split the data into training and test sets
    train_data_index = np.searchsorted(stock.index.values, np.datetime64(start_date))
    train_data = stock.iloc[:train_data_index]
    test_data = stock.loc[start_date:]
    train_data = label_data(train_data)
    test_data = label_data(test_data)

    #trian & test data
    X_train_data = train_data.iloc[0:,:-1]
    y_train_data = train_data.iloc[0:,-1]
    X_test_data = test_data.iloc[0:,:-1]
    y_test_data = test_data.iloc[0:,-1]
    print(len(X_test_data))
    # Normalize the data
    normalizer = MinMaxScaler()
    X_train_data_normalizer = normalizer.fit_transform(X_train_data)
    X_test_data_normalizer = normalizer.transform(X_test_data)

    # # Reshape X_train_data_normalizer
    X_train = X_train_data_normalizer.reshape(X_train_data_normalizer.shape[0], X_train_data_normalizer.shape[1], 1)
    X_test = X_test_data_normalizer.reshape(X_test_data_normalizer.shape[0], X_test_data_normalizer.shape[1], 1)



    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,classification_report,recall_score,precision_recall_curve,f1_score

    # Create a logistic regression model
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)

    # Train the model on the training data
    logistic_model.fit(X_train_data_normalizer, y_train_data)

    # Predict labels for the test set
    y_pred = logistic_model.predict(X_test_data_normalizer)

    # Calculate accuracy
    from sklearn.metrics import accuracy_score,classification_report,recall_score,precision_recall_curve,f1_score,precision_score

    accuracy = accuracy_score(y_test_data, y_pred)
    print("Accuracy:", accuracy)

    print(recall_score(y_test_data, y_pred))
    print(precision_score(y_test_data, y_pred))
    print("f1score ",f1_score(y_test_data, y_pred))
    print(classification_report(y_test_data, y_pred))
    print(y_pred)

    test_data['model_2_sentiment'] = y_pred
    test_data.to_csv('data/dragaon_model_transformer_sentiment.csv')
    return test_data,'data/dragon_model_transformer_sentiment.csv'
