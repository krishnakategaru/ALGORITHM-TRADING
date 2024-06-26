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
warnings.filterwarnings("ignore")



# Define the Transformer model
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = tf.keras.layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

# Build and train the model
input_shape = (7,1)
head_size = 46
num_heads = 60
ff_dim = 55
num_transformer_blocks = 5
mlp_units = [256]
dropout = 0.14
mlp_dropout = 0.4

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    

    for _ in range(num_transformer_blocks):  # This is what stacks our transformer blocks
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)

    outputs = tf.keras.layers.Dense(1, activation="softmax")(x)  # this is a pass-through

    return tf.keras.Model(inputs, outputs)

# Define the learning rate scheduler
def lr_scheduler(epoch, lr, warmup_epochs=30, decay_epochs=100, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr

def fetch_ticker_data(symbol, start_date, end_date):
    """Fetches stock data for a given symbol using yfinance."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(start='2000-01-01', end=end_date)
    return data

def label_data(data):
    # Calculate the percentage change in price from one day to the next
    data['Percentage Change'] = data['Close'].pct_change()
    data['Percentage Change'] = data['Percentage Change'].shift(-1)
    data['Sentiment'] = pd.Series(np.where(data['Percentage Change'] > 0.025, 1, np.where(data['Percentage Change'] < -0.025, -1, 0)), index=data.index)
    # Drop any rows with missing values
    data.dropna(inplace=True)
    data.drop('Percentage Change',axis=1 , inplace=True)
    return data

def train_transformer(symbol_to_fetch,start_date ,end_date,no_model = None):
    #fetching data 
    stock = fetch_ticker_data(symbol_to_fetch, start_date, end_date)
    # Calculate deltas for open, high, low, and close columns
    for i in range(1, 90):  # Calculate deltas up to 5 days
        stock[f"open_delta_{i}day"] = stock["Open"].diff(periods=i)
        stock[f"high_delta_{i}day"] = stock["High"].diff(periods=i)
        stock[f"low_delta_{i}day"] = stock["Low"].diff(periods=i)
        stock[f"close_delta_{i}day"] = stock["Close"].diff(periods=i)
            # Rolling mean and standard deviation of OHLC prices
        stock['Rolling_Mean_Open_{i}day'] = stock['Open'].rolling(window=i).mean()
        stock['Rolling_Mean_High_{i}day'] = stock['High'].rolling(window=i).mean()
        stock['Rolling_Mean_Low_{i}day'] = stock['Low'].rolling(window=i).mean()
        stock['Rolling_Mean_Close_{i}day'] = stock['Close'].rolling(window=i).mean()

        stock['Rolling_Std_Open_{i}day'] = stock['Open'].rolling(window=i).std()
        stock['Rolling_Std_High_{i}day'] = stock['High'].rolling(window=i).std()
        stock['Rolling_Std_Low_{i}day'] = stock['Low'].rolling(window=i).std()
        stock['Rolling_Std_Close_{i}day'] = stock['Close'].rolling(window=i).std()
    stock = stock.fillna(method="ffill", axis=0)
    stock = stock.fillna(method="bfill", axis=0)
    # stock.index = stock.index.date
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
    X_train_data = train_data.iloc[:,:-1]
    y_train_data = train_data.iloc[:,-1]
    X_test_data = test_data.iloc[:,:-1]
    y_test_data = test_data.iloc[:,-1]
    print(len(X_test_data))
    # Normalize the data
    normalizer = MinMaxScaler()
    X_train_data_normalizer = normalizer.fit_transform(X_train_data)
    X_test_data_normalizer = normalizer.transform(X_test_data)

    # # Reshape X_train_data_normalizer
    X_train = X_train_data_normalizer.reshape(X_train_data_normalizer.shape[0], X_train_data_normalizer.shape[1], 1)
    X_test = X_test_data_normalizer.reshape(X_test_data_normalizer.shape[0], X_test_data_normalizer.shape[1], 1)
    
        
    if no_model == 'transformer' :
        model = build_model(
            input_shape,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units=mlp_units,
            mlp_dropout=mlp_dropout,
            dropout=dropout,
        )

        model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["mean_squared_error"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
        ]

        # model.summary()
        history = model.fit(
            X_train,
            y_train_data,
            validation_split=0.2,
            epochs=100,
            batch_size=20,
            callbacks=callbacks,
        )
        model.save('models/transformer_'+f"{symbol_to_fetch}"+"_model.h5")
        model.save('models/transformer_'+f"{symbol_to_fetch}"+"_model.keras")
        no_model = model

    elif no_model == 'svm':
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score,classification_report

        # Create an SVM classifier
        svm_model = SVC(kernel='linear', random_state=42)

        # Train the model on the training data
        svm_model.fit(X_train_data_normalizer, y_train_data)

        # Predict labels for the test set
        y_pred = svm_model.predict(X_test_data_normalizer)
            
        # Calculate accuracy
        accuracy = accuracy_score(y_test_data, y_pred)
        print("Accuracy:", accuracy)
        print(classification_report(y_test_data,y_pred))
        no_model =  svm_model
        return no_model,X_test_data_normalizer,test_data
    elif no_model == 'xgboost':
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import accuracy_score,classification_report,recall_score,precision_recall_curve,f1_score

        # Create an XGBoost classifier
        xgb_model = XGBClassifier(random_state=42,subsample=0.6,gamma = 0.3,colsample_bytree=0.9,reg_lambda = 0.3,reg_alpha = 0,n_estimators= 100,min_child_weight=9,max_depth=3  ,learning_rate = 0.01    )

        # Train the model on the training data
        xgb_model.fit(X_train_data_normalizer, y_train_data)

        # Predict labels for the test set
        y_pred = xgb_model.predict(X_test_data_normalizer)

        # Calculate accuracy
        from sklearn.metrics import accuracy_score,classification_report,recall_score,precision_recall_curve,f1_score,precision_score

        accuracy = accuracy_score(y_test_data, y_pred)
        print("Accuracy:", accuracy)

        print(recall_score(y_test_data, y_pred))
        print(precision_score(y_test_data, y_pred))
        print("f1score ",f1_score(y_test_data, y_pred))
        print(classification_report(y_test_data, y_pred))
        no_model =  xgb_model
        print('returning')
        return no_model,X_test_data_normalizer,test_data

    return no_model,X_test,test_data
    #predictions
def prepare_sentiment_from_transformer(symbol_to_fetch,start_date,end_date,model = None):
    try :
        if  model == 'transformer':
            print("entered")
            model = tf.keras.models.load_model('models/transformer_'+f"{symbol_to_fetch}"+"_model.keras")
            print("passed")
            _,X_test,test_data = train_transformer(symbol_to_fetch = symbol_to_fetch, start_date = start_date, end_date = end_date,no_model=model)
        elif  model == 'svm' :
            model,X_test,test_data  = train_transformer(symbol_to_fetch = symbol_to_fetch, start_date = start_date, end_date = end_date,no_model=model)
        elif  model == 'xgboost' :
            print("entered")
            model,X_test,test_data  = train_transformer(symbol_to_fetch = symbol_to_fetch, start_date = start_date, end_date = end_date,no_model=model)
    except:
        model,X_test,test_data = train_transformer(symbol_to_fetch = symbol_to_fetch, start_date = start_date, end_date = end_date)
    y_pred = model.predict(X_test) # this is the sentiment data 

    test_data['model_1_sentiment'] = y_pred

    test_data.index = test_data.index.date
    test_data.to_csv('data/transformer_sentiment.csv')
    return test_data,'data/transformer_sentiment.csv'
    """next steps :  we need to additionally train the model if model is already present
    or take nearly 30 stocks and train the model with the huge data 
    or take every 5 mins data nad trian with it, and at last mix the test data with day wise"""