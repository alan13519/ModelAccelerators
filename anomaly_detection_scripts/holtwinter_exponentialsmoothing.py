"""
Implementation of Holt Winter's Exponential Smoothing
"""


import pandas as pd
import numpy as np
import statsmodels
import warnings
import os
import scipy.stats as st

from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def mape(y_true, y_pred) -> float:
    """
    Return mean absolute percentage error of predicted vs actual
    :param true: series of the actual values
    :param y_pred: series of the predicted values
    :return: float mean absolute percentage error
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class ExponentialSmoothingClass(object):
    def __init__(self, time_series):

        # Hyper parameter
        hyper_params = {
            'trend': ['add', 'mul'],
            'damped': [True, False],
            'seasonal': ['add', 'mul'],
            'seasonal_periods': [2, 4, 6, 12]
        }
        
        # Set index as 

        # Split into validation set and test set for hyper-tuning purposes
        train_size = int(0.8 * len(time_series))
        train, test = time_series[0:train_size], time_series[train_size:]

        # Train model on default values, we'll use this model as base to compare hyper-tuning models with
        model = ExponentialSmoothing(train)
        model_fit = model.fit()
        y_pred = model_fit.predict(start=len(train), end=len(train) + (len(test) - 1))
        curr_best_mape = mape(test, y_pred)

        # Hyper Tune model, store best parameters into database
        for trend in hyper_params['trend']:
            for damped in hyper_params['damped']:
                for seasonal in hyper_params['seasonal']:
                    for seasonal_period in hyper_params['seasonal_periods']:
                        warnings.filterwarnings("ignore") # Ignore any warnings if there are any
                        model = ExponentialSmoothing(train, trend=trend,
                                                     seasonal=seasonal,
                                                     damped=damped,
                                                     seasonal_periods=seasonal_period,
                                                     freq=train.index.inferred_freq)
                        try:
                            model_fit = model.fit()
                        except:
                            continue
                        y_pred = model_fit.predict(start=len(train), end=len(train) + (len(test) - 1))
                        
                        # Calculate the MAPE score, and get the model with best score
                        curr_mape = mape(test, y_pred)

                        # If current calculated mape is lower than the best, change the variables
                        if curr_mape < curr_best_mape:
                            curr_best_mape = curr_mape
                            best_trend = trend
                            best_damped = damped
                            best_seasonal_periods = seasonal_period
                            best_seasonal = seasonal
                        warnings.filterwarnings("default") # Accept warnings again

        # Train on entire dataset
        model = ExponentialSmoothing(time_series, trend=best_trend,
                                     seasonal=best_seasonal,
                                     damped=best_damped,
                                     seasonal_periods=best_seasonal_periods,
                                     freq=train.index.inferred_freq)
        es = model.fit()

        # Start and end indexes of predictions
        start = 0
        end = start + len(time_series) - 1

        y_pred = model.predict(params=model.params, start=start, end=end)

        time_series = time_series.reset_index()

        time_series['yhat'] = y_pred

        predictions = time_series.copy()

        # Calculate yhat_lower, yhat_upper prediction intervals
        allowed_std_valid = predictions['yhat'].std(axis=0, skipna=True)
        predictions['yhat_upper'] = predictions['yhat'] + (1.96 * allowed_std_valid)
        predictions['yhat_lower'] = predictions['yhat'] - (1.96 * allowed_std_valid)

        forecasted = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y']].copy()

        forecasted['anomaly'] = 0
        forecasted.loc[forecasted['y'] > forecasted['yhat_upper'], 'anomaly'] = 1
        forecasted.loc[forecasted['y'] < forecasted['yhat_lower'], 'anomaly'] = -1

        # anomaly importances
        forecasted['severity'] = ExponentialSmoothingClass.get_severity(forecasted['y'], forecasted['yhat'],
                                                                           forecasted['y'].std())
        forecasted['severity'] = (forecasted['severity'] - forecasted['severity'].min()) / (
                forecasted['severity'].max() - forecasted['severity'].min())

        only_anoms = forecasted[forecasted['anomaly'] != 0]
        only_anoms['prediction'] = only_anoms['yhat']
        only_anoms.drop(['yhat'], axis=1, inplace=True)

        self.anoms = only_anoms

    @staticmethod
    def get_severity(actual, forecast, stdev):
        z_score = abs((actual - forecast) / stdev)
        severity = st.norm.cdf(z_score)
        return severity

# We need this because we need to convert the date back into the month
def find_month_index(x, dictionary):
    return dictionary[x]
