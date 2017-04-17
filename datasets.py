from __future__ import division, print_function


import utils
import pandas as pd
import numpy as np
import os
import env


def filename(dataset, lookahead, n_periods, sym, year, include_path=True):
    fname = "{}-LA{:03d}-W{:03d}-{}-{}.h5".format(dataset, lookahead, n_periods, sym, year)

    if include_path:
        return os.path.join(env.input_dir(), fname)
    else:
        return fname


def save(X, Y, prices, filename):
    """
    saves the given feature dataframe, label series and price series to the specified file
    :param X: features design matrix as pandas Dataframe
    :param Y: labels as pandas Series
    :param prices: price series, as pandas Series.
    :param filename: filename into which to save
    """
    with pd.HDFStore(filename, 'w') as store:
        X.to_hdf(store, 'X')
        Y.to_hdf(store, 'Y')
        prices.to_hdf(store, 'prices')


def load(filename):
    """
    loads feature matrix, label series and price series from the specified file.
    :param filename: file from which to load the data
    :return: tuple (X,Y,prices)
        X - feature matrix as pandas DataFrame
        Y - label series as pandas Series
        prices - price series as pandas Series
    """
    X = pd.read_hdf(filename, 'X')
    Y = pd.read_hdf(filename, 'Y')
    prices = pd.read_hdf(filename, 'prices')
    return X, Y, prices


def prepare_dataset1(df, lookahead, window):
    """
    prepares a dataset consisting of
    X (features):
        - ohlc relative to current close over a window of lookback periods
    Y (labels):
        - binary (0,1) categories, corresponding to direction of future return over *lookahead* periods
    prices:
        - closing price series, with index aligned to the features and labels

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

    return X, Y, px


def prepare_dataset2(df, lookahead, n_periods):
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
    Note that NaN's are dropped. and that features and labels are normalised based on an exponentially weighted
    moving mean and standard deviation, over a window that's 100 times bigger than *window*.

    :param df: dataframe of open, high, low, close price data with datetime index.
    :param lookahead: number of periods to look ahead to compute the future return direction label.
    :param n_periods: total number of periods of history to create futures for.
    :return: X,Y,prices
    """

    px = df['close']

    max_lb_idx = max(0, n_periods - 4)
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

    return X, Y, px


def prepare_dataset3(df, lookahead, n_periods, lookback_spacing="squared"):
    """
    Differs from dataset2, in the treatment of open, high, low historic features, which are computed relative to the close at that time, not the latest close.
    
    :param df: dataframe of open, high, low, close price data with datetime index.
    :param lookahead: number of periods to look ahead to compute the future return direction label.
    :param n_periods: total number of periods of history to create futures for.
    :return: X,Y,prices
    """
    df = resample_and_forward_fill(df, 10)
    px = df['close']

    if lookback_spacing == "squared":
        lookbacks = lookbacks_squared(n_periods)
    elif lookback_spacing == "cubed":
        lookbacks = lookbacks_cubed(n_periods)
    else:
        raise ValueError("unexpected lookback spacing type: " + lookback_spacing)


    X = pd.DataFrame(index=df.index)
    for i in lookbacks:
        period_i_close = df['close'].shift(i)
        X['open-{}'.format(i)] = (df['open'].shift(i) / period_i_close) - 1
        X['high-{}'.format(i)] = (df['high'].shift(i) / period_i_close) - 1
        X['low-{}'.format(i)] = (df['low'].shift(i) / period_i_close) - 1

        # don't add close-0, as it's always zero
        if i > 0:
            X['close-{}'.format(i)] = (df['close'].shift(i) / px) - 1

    # Normalise features, by removing long-term mean components and scaling to acheive a std-deviation of approx 1.
    # since this is timeseries, data we shouldn't consider the whole data set, just data in the past, as of any
    # given moment. Hence we use rolling measures of mean and std. This also has the advantage that
    # use Exponentially weighted moving averages, since they are smoother than simple moving averages.

    # base the normalisation on 10 times the largest lookback
    NORMALISATION_WINDOW = lookbacks[-1]*10
    X = X-X.ewm(com=NORMALISATION_WINDOW, min_periods=NORMALISATION_WINDOW, ignore_na=True).mean()
    # replace zero standard deviation values with NaN. This will ensure result is also a NaN, rather than infinite.
    X = X/X.ewm(com=NORMALISATION_WINDOW, min_periods=NORMALISATION_WINDOW, ignore_na=True).std().replace(0, np.nan)

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

    return X, Y, px


def lookbacks_squared(n_periods):
    max_lb_idx = max(0, n_periods - 4)
    lookbacks = np.concatenate([np.arange(4), 4+(np.arange(max_lb_idx) **2)]).astype(int)
    return lookbacks

def lookbacks_cubed(n_periods):
    max_lb_idx = max(0, n_periods - 4)
    lookbacks = np.concatenate([np.arange(4), 4+(np.arange(max_lb_idx) **3)]).astype(int)
    return lookbacks

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


def fractional_time_of_week(ts):
    """ converts a pandas Timestamp into a float representing the relative time within a calendar week 0 and 2*pi """
    return 2*np.pi * (ts.day*24*60  + ts.hour*60 + ts.minute)/(24*60*7)

def fractional_time_of_day(ts):
    """ converts a pandas Timestamp into a float representing the time of day between 0 and 2*pi """
    return 2*np.pi * (ts.hour*60 + ts.minute)/(24*60)

def resample_and_forward_fill(df, maxfill=0):
    """
    Resamples the specified OHLC dataframe to 1 minute samples, forward filling for 'maxfill' periods.
    Note that open, high and low columns get forward filled from the previous close sample
    """
    df = df.resample('1T', label='right').asfreq()

    # open, high and low get forward filled from close
    df['close'] = df['close'].ffill(limit=maxfill)
    df.loc[df['open'].isnull(), 'open'] = df['close']
    df.loc[df['high'].isnull(), 'high'] = df['close']
    df.loc[df['low'].isnull(), 'low'] = df['close']

    return df


if __name__ == '__main__':

    df = utils.load_1minute_fx_bars("EURSEK", 2012)
    df = resample_and_forward_fill(df)
    #df = random_ohlc(200000)

    X_train, Y_train, price_train = prepare_dataset3(df[:100000], lookahead=1, n_periods=3)
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
    hist = model.fit(X_train.as_matrix(), Y_train, max_epochs=20)

    X_test, Y_test, price_test = prepare_dataset3(df[100000:200000], lookahead=1, n_periods=3)
    Y_pred = model.predict(X_test.as_matrix())

    import metrics
    metrics.performance_report("foo", price_test, lookahead, Y_test, Y_pred)

