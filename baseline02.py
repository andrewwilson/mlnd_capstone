import pandas as pd
import numpy as np

"""

Last Price Change Predictor

Implementation of a baseline price prediction model which simply predicts the next price, as the last price
plus the most recent price change.

"""

def predict(price_series, lookahead):
    """
    :param price_series: the historic price series, as a pandas Series
    :param lookahead:
    :return: prediction of the future price, lookahead periods into the future at time t.
    """
    diffs = price_series - price_series.shift(lookahead)
    return price_series + diffs

