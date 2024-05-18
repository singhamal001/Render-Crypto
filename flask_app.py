# app.py
from flask import Flask, request, jsonify
import ccxt
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)

# Ensure reproducibility
np.random.seed(1)

exchange = ccxt.kucoin()

def fetch_data(ticker, timeframe):
    ticker_ohlcv = exchange.fetch_ohlcv(ticker, timeframe=timeframe, limit=2000)
    df = pd.DataFrame(ticker_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = df['timestamp'].apply(lambda ts: datetime.datetime.fromtimestamp(ts / 1000))
    df.drop('timestamp', axis=1, inplace=True)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    df['target'] = (df['log_return'] > 0).astype(int)
    unscaled_df = df[['datetime', 'open', 'high', 'low', 'close']].copy()
    scaler = StandardScaler()
    df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    return df, unscaled_df

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC()
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
    return results

@app.route('/predict', methods=['POST'])
def train():
    data = request.get_json()
    ticker = data['ticker']
    timeframe = data['timeframe']

    df, unscaled_df = fetch_data(ticker, timeframe)
    X = df[['open', 'high', 'low', 'close', 'volume', 'log_return']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    latest_candle = df.iloc[-1]
    predictions = {}
    for name, model in models.items():
        prediction = model.predict(latest_candle[['open', 'high', 'low', 'close', 'volume', 'log_return']].values.reshape(1, -1))
        predictions[name] = int(prediction[0])

    price_data = unscaled_df.tail(50).to_dict(orient='list')

    return jsonify({
        'results': results,
        'latest_candle_prediction': predictions,
        'price_data': price_data
    })

if __name__ == '__main__':
    app.run(debug=True,host = '0.0.0.0', port=5000)
