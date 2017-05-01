import pandas as pd
import numpy as np

"""

Last Price Change Predictor

Implementation of a baseline price prediction model which simply predicts the next period's price change
as the same as the last period's price change.

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
        :return: prediction of the future price change, "lookahead" periods into the future at time t.
        """

        # compute recent price changes by looking back over the same number of periods that we wish to predict ahead.
        changes = price_series - price_series.shift(self.lookahead)

        # predict future change as same as last change.

        # center predictions around 0.5 rather than 0, for consistency with our neural network models
        return changes.fillna(0.0) + 0.5
