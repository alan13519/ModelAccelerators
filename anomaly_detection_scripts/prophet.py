"""
Implementation of Prophet for Single Point Anomaly detection
"""

import os
import pandas as pd
from fbprophet import Prophet
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import scipy.stats as st

class FBProphet(object):
    """
    Facebook's Prophet time series prediction, used to predict anomalies
    """

    def __init__(self, time_series, seasonality_mode='additive', period='month', seasonality_name='quarterly', fourier_order=5):
        """
        Initialization
        :param time_series: uni-variate time series data
        :param seasonality_mode: Additive or Multiplicative seasonlity mode 
        :param period: Period of time series, can be 'month', 'day', 'year'
        """
        time_series = time_series.reset_index()  # Seperate the single column series into two

        time_series.rename(columns={time_series.columns[0]: "ds"}, inplace=True)
        if not is_datetime(time_series.ds.dtype):
            print("===============================")
            print("Warning column conversion error")
            print("===============================")
        time_series.rename(columns={time_series.columns[1]: "y"}, inplace=True)

        if period == 'month':
            period_length = float(365 / 12)
        elif period == 'day':
            period_length = 365
        else:
            # We can add more later, but for now only month or day
            raise Exception("Error, period must be 'month' or 'day'")

        # Prophet model
        model = Prophet(seasonality_mode=seasonality_mode, interval_width=0.95)  # 95% Confidence Interval
        model.add_seasonality(name=seasonality_name, period=3, fourier_order=fourier_order)
        model_fit = model.fit(time_series)

        # Prediction
        future = model_fit.make_future_dataframe(periods=int(len(time_series) * period_length), freq='d')

        predictions = model_fit.predict(future)
        predictions['y'] = time_series['y'].reset_index(drop=True)

        forecasted = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y']].copy()
        forecasted['anomaly'] = 0
        forecasted.loc[forecasted['y'] > forecasted['yhat_upper'], 'anomaly'] = 1
        forecasted.loc[forecasted['y'] < forecasted['yhat_lower'], 'anomaly'] = -1
        # anomaly importances
        forecasted['severity'] = FBProphet.get_severity(forecasted['y'], forecasted['yhat'], forecasted['y'].std())
        forecasted['severity'] = (forecasted['severity'] - forecasted['severity'].min()) / (forecasted['severity'].max() - forecasted['severity'].min())
        
        only_anoms = forecasted[forecasted['anomaly'] != 0]
        only_anoms['prediction'] = only_anoms['yhat']
        only_anoms.drop(['yhat'], axis=1, inplace=True)

        self.anoms = only_anoms

    @staticmethod
    def get_severity(actual, forecast, stdev):
        z_score = abs((actual - forecast) / stdev)
        severity = st.norm.cdf(z_score)
        return severity
