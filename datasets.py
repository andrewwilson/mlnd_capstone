from __future__ import division, print_function


import utils
import pandas as pd
import numpy as np
import os
import env


def filename(dataset, lookahead, window, sym, year, include_path=True):
    fname = "{}-LA{:03d}-W{:03d}-{}-{}.h5".format(dataset, lookahead, window, sym, year)

    if include_path:
        return os.path.join(env.input_dir(), fname)
    else:
        return fname


def save(X, Y, prices, future_returns, filename):
    """
    saves the given feature dataframe, label series and price series to the specified file
    :param X: features design matrix as pandas Dataframe
    :param Y: labels as pandas Series
    :param prices: price series, as pandas Series.
    :param future_returns: as pandas Seriers
    :param filename: filename into which to save
    """
    with pd.HDFStore(filename, 'w') as store:
        X.to_hdf(store, 'X')
        Y.to_hdf(store, 'Y')
        prices.to_hdf(store, 'prices')
        future_returns.to_hdf(store, 'future_returns')


def load(filename):
    """
    loads feature matrix, label series and price series from the specified file.
    :param filename: file from which to load the data
    :return: tuple (X,Y,prices)
        X - feature matrix as pandas DataFrame
        Y - label series as pandas Series
        prices - price series as pandas Series
        future_returns - series of futures returns, as pandas Series
    """
    X = pd.read_hdf(filename, 'X')
    Y = pd.read_hdf(filename, 'Y')
    prices = pd.read_hdf(filename, 'prices')
    future_returns = pd.read_hdf(filename, 'future_returns')
    return X, Y, prices, future_returns


def prepare_dataset1(df, lookahead, window):
    """
    prepares a dataset consisting of
    X (features):
        - ohlc relative to current close over a window of lookback periods
    Y (labels):
        - binary (0,1) categories, corresponding to direction of future return over *lookahead* periods
    prices:
        - closing price series, with index aligned to the features and labels
    future_returns
        - series of future returns of the closing price series, lookahead periods into the future
    Note that NaN's are dropped. and that features and labels are normalised based on an exponentially weighted
    moving mean and standard deviation, over a window that's 100 times bigger than *window*.

    :param df: dataframe of open, high, low, close price data with datetime index.
    :param lookahead: number of periods to look ahead to compute the future return direction label.
    :param window: total number of periods of history to create futures for.
    :return: X,Y,prices,future_returns
    """

    px = df['close']

    X = pd.DataFrame(index=df.index)
    for i in range(window + 1):
        X['open-{}'.format(i)] = (df['open'].shift(i) / px) - 1
        X['high-{}'.format(i)] = (df['high'].shift(i) / px) - 1
        X['low-{}'.format(i)] = (df['low'].shift(i) / px) - 1

        # don't add close-0, as it's always zero
        if i > 0:
            X['close-{}'.format(i)] = (df['close'].shift(i) / px) - 1

    # Normalise features, by removing long-term mean components and scaling to acheive a std-deviation of approx 1.
    # since this is timeseries, data we shouldn't consider the whole data set, just data in the past, as of any
    # given moment. Hence we use rolling measures of mean and std. This also has the advantage that
    # use Exponentially weighted moving averages, since they are smoother than simple moving averages.
    NORMALISATION_WINDOW=window*100
    X = X-X.ewm(com=NORMALISATION_WINDOW, min_periods=NORMALISATION_WINDOW).mean()
    X = X/X.ewm(com=NORMALISATION_WINDOW, min_periods=NORMALISATION_WINDOW).std()

    fut_ret = utils.future_return(px, lookahead)
    # normalise future return used for categorisation, by subtracting rolling mean.
    # note that we return the raw, un-normalised future return.
    normed_fut_ret = fut_ret - fut_ret.ewm(com=NORMALISATION_WINDOW).mean()
    Y = utils.categoriser2(normed_fut_ret)

    # drop any records which are null in either X or y
    idx_x = X.dropna().index
    idx_y = Y.dropna().index
    idx = idx_x.intersection(idx_y)

    X = X.ix[idx]
    Y = Y.ix[idx]
    px = px.ix[idx]
    fut_ret = fut_ret.ix[idx]

    return X, Y, px, fut_ret


def prepare_dataset2(df, lookahead, n_features):
    """
    Differs from dataset1, in that the window is not simply n previous periods, but covers a greater time period
    aimed at making more efficient use of the number of features

    Uses every period for first 5 periods, then expands the range as the index squared, to cover over 400 timesteps
    with just 25 feature indices.
    [  0   1   2   3   4   5   8  13  20  29  40  53  68  85
      104 125 148 173 200 229 260 293 328 365 404]

    prepares a dataset consisting of
    X (features):
        - ohlc relative to current close over a window of lookback periods
    Y (labels):
        - binary (0,1) categories, corresponding to direction of future return over *lookahead* periods
    prices:
        - closing price series, with index aligned to the features and labels
    future_returns
        - series of future returns of the closing price seriers, lookahead periods into the future
    Note that NaN's are dropped. and that features and labels are normalised based on an exponentially weighted
    moving mean and standard deviation, over a window that's 100 times bigger than *window*.

    :param df: dataframe of open, high, low, close price data with datetime index.
    :param lookahead: number of periods to look ahead to compute the future return direction label.
    :param n_features: total number of periods of history to create futures for.
    :return: X,Y,prices,future_returns
    """

    px = df['close']

    max_lb_idx = max(0, n_features - 4)
    lookbacks = np.concatenate([np.arange(4), 4+(np.arange(max_lb_idx) **2)]).astype(int)

    X = pd.DataFrame(index=df.index)
    for i in lookbacks:
        X['open-{}'.format(i)] = (df['open'].shift(i) / px) - 1
        X['high-{}'.format(i)] = (df['high'].shift(i) / px) - 1
        X['low-{}'.format(i)] = (df['low'].shift(i) / px) - 1

        # don't add close-0, as it's always zero
        if i > 0:
            X['close-{}'.format(i)] = (df['close'].shift(i) / px) - 1

    # Normalise features, by removing long-term mean components and scaling to acheive a std-deviation of approx 1.
    # since this is timeseries, data we shouldn't consider the whole data set, just data in the past, as of any
    # given moment. Hence we use rolling measures of mean and std. This also has the advantage that
    # use Exponentially weighted moving averages, since they are smoother than simple moving averages.

    # base the normalisation on 10 times the largest lookback
    NORMALISATION_WINDOW = lookbacks[-1]*10
    X = X-X.ewm(com=NORMALISATION_WINDOW, min_periods=NORMALISATION_WINDOW).mean()
    X = X/X.ewm(com=NORMALISATION_WINDOW, min_periods=NORMALISATION_WINDOW).std()

    fut_ret = utils.future_return(px, lookahead)
    # normalise future return used for categorisation, by subtracting rolling mean.
    # note that we return the raw, un-normalised future return.
    normed_fut_ret = fut_ret - fut_ret.ewm(com=NORMALISATION_WINDOW).mean()
    Y = utils.categoriser2(normed_fut_ret)

    # drop any records which are null in either X or y
    idx_x = X.dropna().index
    idx_y = Y.dropna().index
    idx = idx_x.intersection(idx_y)

    X = X.ix[idx]
    Y = Y.ix[idx]
    px = px.ix[idx]
    fut_ret = fut_ret.ix[idx]

    return X, Y, px, fut_ret


def random_series(n, mean=100, std=0.00033, seconds=10):
    """
    returns a random series, as an unpredicable price series
    :param n:
    :param mean:
    :param std:
    :return:
    """
    s = (np.random.randn(n) * std).cumsum() + mean

    idx = pd.DatetimeIndex(np.arange(len(s)) * 1e9* seconds)
    return pd.Series(s, index=idx)


def random_ohlc(n):
    """
    returns an open-high-low-close resampling of a random price series
    :param n: number of 1 minute samples
    :return:
    """
    s = random_series(n * 6, seconds=10)
    return s.resample('1min').agg({'open': lambda x: x[0], 'high': max, 'low': min, 'close': lambda x: x[-1]})


if __name__ == '__main__':

    df = utils.load_1minute_fx_bars("EURSEK", 2009)
    X_train, Y_train, price_train, _ = prepare_dataset1(df[:100000], lookahead=1, window=3)
    print(X_train.describe().transpose())
    print(Y_train[:20])

    print (X_train.shape)
    n_features = X_train.shape[1]
    print ("n_features:", n_features)

    lookahead = 1
    n_categories = 2
    layer_widths = [10,10,10]
    dropout = 0

    from model01 import MLPModel01
    model = MLPModel01(lookahead, n_features, n_categories, layer_widths, dropout)
    print (model.summary())
    #
    hist = model.fit(X_train.as_matrix(), Y_train)

    X_test, Y_test, price_test, fut_returns_test = prepare_dataset1(df[100000:200000], lookahead=1, window=3)
    Y_pred = model.predict(X_test.as_matrix())

    fut_returns_test = fut_returns_test - fut_returns_test.mean()
    #print (utils.prediction_to_category2(Y_pred) * fut_returns_test)
