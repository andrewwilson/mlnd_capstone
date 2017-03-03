from __future__ import division, print_function
from sklearn.metrics import f1_score

from model01 import MLPModel01
import utils
import pandas as pd
import numpy as np

#
# def logreturn(px_latest, px_prev):
#     return np.log(px_latest / px_prev)
#
# def normalise_price(p, vol_adjust=None):
#     px = logreturn(p, p.shift(1)).cumsum()
#     if vol_adjust:
#         px = px / ewm_vol(px, com=vol_adjust)
#     return px
#
# def ewm_vol(p,com):
#     return p.diff().ewm(com, min_periods=com).std()


def prepare_data(df, lookahead, window):

    px = df['close']

    X = pd.DataFrame(index=df.index)
    for i in range(window + 1):
        X['open-{}'.format(i)] = (df['open'].shift(i) / px) - 1
        X['high-{}'.format(i)] = (df['high'].shift(i) / px) - 1
        X['low-{}'.format(i)] = (df['low'].shift(i) / px) - 1

        # don't add close-0, as it's always zero
        if i > 0:
            X['close-{}'.format(i)] = (df['close'].shift(i) / px) - 1

    # Normalise features, by removing long-term mean components and scaling to std-deviation of approx 1.
    # since this is timeseries, data we shouldn't consider the whole data set, just data in the past, as of any
    # given moment. Hence we use rolling measures of mean and std.
    # use Exponentially weighted moving averages, since they are smoother than simple moving averages.
    NORMALISATION_WINDOW=window*100
    X = X-X.ewm(com=NORMALISATION_WINDOW).mean()
    X = X/X.ewm(com=NORMALISATION_WINDOW).std()

    # normalise future return used for categorisation, by subtracting rolling mean.
    fut_ret = utils.future_return(px, lookahead)
    fut_ret = fut_ret - fut_ret.ewm(com=NORMALISATION_WINDOW).mean()
    Y = utils.categoriser2(fut_ret)

    # TODO: drop any records which are null in either X or y
    # for now, lets just fill as zero

    X = X.fillna(0)
    Y = np.nan_to_num(Y)

    return X, Y, px



if __name__ == '__main__':
    df = utils.load_1minute_fx_bars("USDJPY", 2009)
    X_train, Y_train, price_train = prepare_data(df[:100000], lookahead=1, window=3)
    print(X_train.describe().transpose())
    print(Y_train[:20])

    print (X_train.shape)
    n_features = X_train.shape[1]
    print ("n_features:", n_features)

    lookahead = 1
    n_categories = 2
    layer_widths = [10,10,10]
    dropout = 0

    model = MLPModel01(lookahead, n_features, n_categories, layer_widths, dropout)
    print (model.summary())
    #
    hist = model.fit(X_train.as_matrix(), Y_train, validation_split=0.1)

    X_test, Y_test, price_test = prepare_data(df[100000:200000], lookahead=1, window=3)
    Y_pred = model.model.predict(X_test)
