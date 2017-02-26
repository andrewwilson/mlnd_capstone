import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize_scalar


"""

Moving Average Price Change Predictor

Implementation of a baseline price prediction model which predicts the next price, as the last price
plus the recent moving average price change.

2 Variants are provided:
- Baseline02SMA - which uses a simple moving average
- Baseline02EWMA - which uses an exponentially weighted moving averae

"""

class Baseline02SMA:

    def __init__(self, window, lookahead):
        """
        :param window: the size of the window of the simple moving average function that the model uses.
        :param lookahead: the number of periods ahead which the model should predict
        """
        self.window = window
        self.lookahead = lookahead

    def predict(self, price_series):
        """
        :param price_series: the historic price series, as a pandas Series
        :param lookahead:
        :return: prediction of the future price, lookahead periods into the future at time t.
        """
        # create a price change series, looking back the same number of periods that we wish to predict forwards.
        price_changes = price_series - price_series.shift(self.lookahead)

        # construct a moving average price change
        ma_price_change = price_changes.rolling(self.window).mean()

        return price_series + ma_price_change


class Baseline02EWMA:

    def __init__(self, centre_of_mass, lookahead):
        """
        :param com: the centre of mass of the Exponentially Weighted Moving Average to use
        :param lookahead: the number of periods ahead which the model should predict
        """
        self.centre_of_mass = centre_of_mass
        self.lookahead = lookahead

    def predict(self, price_series):
        """
        :param price_series: the historic price series, as a pandas Series

        :return: prediction of the future price, lookahead periods into the future at time t.
        """
        # create a price change series, looking back the same number of periods that we wish to predict forwards.
        price_changes = price_series - price_series.shift(self.lookahead)

        # construct a moving average price change
        ma_price_change = price_changes.ewm(com=self.centre_of_mass).mean()

        return price_series + ma_price_change

    # def fit(self, price_series, n_iterations=20, metric_fn=mean_squared_error):
    #     """
    #     tune the model's parameters to give the best prediction on the price series
    #     """
    #
    #     px = price_series-price_series.mean()
    #
    #     def loss_fn(c):
    #         self.centre_of_mass = c
    #         prediction = self.predict(px)
    #         truth = px.shift(-self.lookahead)
    #
    #         y_true = np.sign(fut_return.fillna(0)).astype(int)
    #         y_pred = np.sign(pred_return.fillna(0)).astype(int)
    #
    #         # align series
    #         df= pd.DataFrame({'truth':truth, 'prediction':prediction})
    #         df.dropna(inplace=True)
    #         loss = metric_fn(df['truth'], df['prediction'])
    #         print c, loss
    #         return loss
    #     foo = minimize_scalar(loss_fn, options={'maxiter':n_iterations})
    #     print foo


