import pandas as pd
import numpy as np

import keras
from keras import optimizers
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('always')

class LSTM_Class:
    def __init__(self, time_series, epochs=10, time_step=30, drop_out=0.2,
                 num_nodes=64, activation='relu', loss_func='mse',
                optimizer='adam', batch_size=35, validation_split=0.25,
                allowed_std_rate=1.5):
        """
        LSTM Autoencoder for Anomaly Detection
        higher epochs/nodes or low time steps may lead to overfitting,
        """
        # Save time index here
        time_index = time_series.index
        
        # original data
        og_data = time_series.values
        
        time_series = np.expand_dims(time_series, 1)
        
        # Scale it here
        scaler = StandardScaler()
        scaler = scaler.fit(time_series)
        time_series = scaler.transform(time_series)
        
        # We are encoding the same sequence so X and y are the same
        X_train, y_train = LSTM_Class.create_dataset(time_series, time_series, time_step)
        
        n_features = X_train.shape[2] # Used for LSTM input and output
        
        # LSTM Autoencoder 
        model = Sequential()   
        model.add(LSTM(num_nodes, activation=activation, input_shape=(time_step, n_features)))
        model.add(Dropout(drop_out))
        model.add(RepeatVector(time_step))
        model.add(LSTM(num_nodes, activation=activation, return_sequences=True))
        model.add(Dropout(drop_out))
        model.add(TimeDistributed(Dense(n_features)))
        
        model.compile(loss=loss_func, optimizer=optimizer)
        
        history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, 
                            validation_split=validation_split, shuffle=False)
                 
        pred = model.predict(X_train)
                 
        # Unravel LSTM output back into original format (without date index)
        predictions = LSTM_Class.combine_seq(pred, scaler) 
                 
        # Create dataframe
        predictions = pd.DataFrame({'yhat':predictions}, index=time_index.set_names('ds')).reset_index()
        predictions['y'] = og_data
                 
        # Calculate yhat_lower, yhat_upper prediction intervals
        allowed_std_valid = predictions['yhat'].std(axis=0, skipna=True)
        predictions['yhat_upper'] = predictions['yhat'] + (allowed_std_rate * allowed_std_valid)
        predictions['yhat_lower'] = predictions['yhat'] - (allowed_std_rate * allowed_std_valid)
        
        forecasted = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y']].copy()

        forecasted['anomaly'] = 0
        forecasted.loc[forecasted['y'] > forecasted['yhat_upper'], 'anomaly'] = 1
        forecasted.loc[forecasted['y'] < forecasted['yhat_lower'], 'anomaly'] = -1
                 
        # anomaly importances
        forecasted['severity'] = LSTM_Class.get_severity(forecasted['y'], forecasted['yhat'],
                                                                           forecasted['y'].std())
        forecasted['severity'] = (forecasted['severity'] - forecasted['severity'].min()) / (
                forecasted['severity'].max() - forecasted['severity'].min())

        only_anoms = forecasted[forecasted['anomaly'] != 0].copy()
        only_anoms['prediction'] = only_anoms['yhat']
        only_anoms.drop(['yhat'], axis=1, inplace=True)
        
        self.forecasted = forecasted
        self.anoms = only_anoms
    
    # plot the data (Jupyter Notebook)
    def plot_data(self, figsize=(20,15), style='Solarize_Light2'):
        plt.style.use(style)
        plt.figure(figsize=figsize)
        
        # Hide the dates
        plt.gca().get_xaxis().set_visible(False)
        
        plt.plot(self.forecasted.set_index('ds')['y'])
        # Un/comment the bottom two lines to see the upper and lower bounds
        plt.plot(self.forecasted['yhat_upper'], color='r', alpha=0.5, linestyle='-')
        plt.plot(self.forecasted['yhat_lower'], color='r', alpha=0.5, linestyle='-')
        plt.scatter(self.anoms['ds'], self.anoms.set_index('ds')['y'], color='r', marker='o', s=200)
        plt.show()
    
    # Create LSTM batches
    @staticmethod
    def create_dataset(X, y, time_steps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X[i:(i + time_steps)]
            Xs.append(v)
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)
                 
                 
    @staticmethod
    def combine_seq(time_series, scaler):
        """
        Unravels sequence data back to original
        """
        seq = time_series.shape[0]
        time_step = time_series.shape[1]
        total_output = []
        inter = 0
        for i in range(0, seq+time_step):
            seq_output = []
            if i < seq:
                for j in range(0, min(i+1, time_step)):
                    if i > time_step:
                        var = i - time_step + j
                        var2 = time_step-1-j
                        if i == time_step:
                            var2 -= 1
                    else:
                        var = j
                        var2 = i - j
                        if i == time_step:
                            var2 -= 1
                    seq_output.append(time_series[var].squeeze()[var2])
                inter += 1
            else:
                var = i - time_step
                inter += 1
                for j in range(time_step - 1, (i - seq)-1, -1):
                    seq_output.append(time_series[var].squeeze()[j])
                    var += 1

            total_output.append(scaler.inverse_transform([np.mean(seq_output)])[0])
        return total_output
                 
    @staticmethod
    def get_severity(actual, forecast, stdev):
        z_score = abs((actual - forecast) / stdev)
        severity = st.norm.cdf(z_score)
        return severity