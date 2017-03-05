from __future__ import division, print_function

from model01 import MLPModel01
import utils
import pandas as pd


def save(X, Y, prices, filename):
    """
    saves the given feature dataframe, label series and price series to the specified file
    :param X: features design matrix as pandas Dataframe
    :param Y: labels as pandas Series
    :param prices: price series, as pandas Series.
    :param filename: filename into which to save
    """
    store = pd.HDFStore(filename, 'w')
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
    :return: X,Y, prices
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

    # normalise future return used for categorisation, by subtracting rolling mean.
    fut_ret = utils.future_return(px, lookahead)
    fut_ret = fut_ret - fut_ret.ewm(com=NORMALISATION_WINDOW).mean()
    Y = utils.categoriser2(fut_ret)

    # drop any records which are null in either X or y
    idx_x = X.dropna().index
    idx_y = Y.dropna().index
    idx = idx_x.intersection(idx_y)

    X = X.ix[idx]
    Y = Y.ix[idx]
    px = px.ix[idx]

    return X, Y, px


if __name__ == '__main__':
    df = utils.load_1minute_fx_bars("USDJPY", 2009)
    X_train, Y_train, price_train = prepare_dataset1(df[:100000], lookahead=1, window=3)
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

    X_test, Y_test, price_test = prepare_dataset1(df[100000:200000], lookahead=1, window=3)
    Y_pred = model.model.predict(X_test)
