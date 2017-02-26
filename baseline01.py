import pandas as pd
import numpy as np

"""

Last Price Change Predictor

Implementation of a baseline price prediction model which simply predicts the next price, as the last price
plus the most recent price change.

"""

class Baseline01:
    def __init__(self, lookahead):
        """
            :param lookahead: the number of periods ahead which the model should predict
        """
        self.lookahead = lookahead

    def predict(self, price_series):
        """
        :param price_series: the historic price series, as a pandas Series
        :param lookahead:
        :return: prediction of the future price, "lookahead" periods into the future at time t.
        """

        # compute recent price changes by looking back over the same number of periods that we wish to predict ahead.
        diffs = price_series - price_series.shift(self.lookahead)
        return price_series + diffs

