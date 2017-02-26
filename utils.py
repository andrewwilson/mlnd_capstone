import pandas as pd
import numpy as np
import os, re, collections
from collections import namedtuple
from functools import partial
from sklearn.metrics import mean_squared_error
from env import FX_DATASET_DIR

def rmse_rel_std(y_true, y_predict):
    return np.sqrt(mean_squared_error(y_true, y_predict))/y_true.std()

def split_dataset(ds, *sizes):
    """ splits a dataframe into parts of relative size given by: sizes."""

    total_size = sum(sizes)
    total_items = len(ds)

    n_items = [round(total_items * s / total_size) for s in sizes]

    # compensate for rounding by adjusting the largest item
    largest_idx = n_items.index(max(n_items))
    n_items[largest_idx] += total_items - sum(n_items)

    assert sum(n_items) == total_items

    results = []
    last_idx = 0
    for n in n_items:
        idx = int(last_idx + n)
        results.append(ds[last_idx:idx])
        last_idx = idx

    return results



def load_stock_data(sym):
    df = pd.read_csv('stockData/{sym}.csv'.format(sym=sym), index_col=0, parse_dates=True)
    df['px'] = df['Adjusted Close']
    del df['Adjusted Close']
    return df


def random_price_series(samples, std):
    px = pd.Series(((np.random.randn(samples) * std) + 1).cumprod(), name='px')
    return pd.DataFrame(px)


DataSet = namedtuple('DataSet', ['name', 'X_train', 'Y_train', 'X_dev', 'Y_dev', 'X_test', 'Y_test'])


def fx_1minute_bar_catalog():
    res = collections.defaultdict(list)
    files = os.listdir(FX_DATASET_DIR)
    for f in files:
        m = re.search('DAT_ASCII_(\w+)_M1_(\d+).csv',f)
        if m:
            sym, date = m.groups()
            res[sym].append(date)
    return res

def load_1minute_fx_bars(sym, date):
    filename = os.path.join(FX_DATASET_DIR, 'DAT_ASCII_{sym}_M1_{date}.csv'.format(sym=sym, date=date))
    df = pd.read_csv(filename, header=None, sep=';',
                     names = ['ts', 'open','high', 'low','close','volume']
                    )
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True, verify_integrity=True)
    return df

def load_stock_datasets(features_and_targets_fn, train_frac=75, dev_frac=15, test_frac=15,
                        sym_filter_fn=lambda x: True):
    res = {}
    files = os.listdir('stockData')
    for f in files:
        sym = f.split('.')[0]
        if sym_filter_fn(sym):
            raw = load_stock_data(sym)
            train, dev, test = split_dataset(raw, train_frac, dev_frac, test_frac)

            X_train, Y_train = features_and_targets_fn(train)
            X_dev, Y_dev = features_and_targets_fn(dev)
            X_test, Y_test = features_and_targets_fn(test)

            ds = DataSet(sym, X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
            res[sym] = ds
    return res


def create_random_datasets(features_and_targets_fn,
                           samples=6000, train_frac=75, dev_frac=15, test_frac=15,
                           syms_to_std_map={'S1': 0.015}):
    res = {}
    for sym in syms_to_std_map.keys():
        std = syms_to_std_map[sym]
        raw = random_price_series(samples=samples, std=std)
        train, dev, test = split_dataset(raw, train_frac, dev_frac, test_frac)

        X_train, Y_train = features_and_targets_fn(train)
        X_dev, Y_dev = features_and_targets_fn(dev)
        X_test, Y_test = features_and_targets_fn(test)

        ds = DataSet(sym, X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
        res[sym] = ds
    return res


def logreturn(px_latest, px_prev):
    return np.log(px_latest / px_prev)


def FT_logreturn_vs_logreturn(df, return_lookbacks=[1], target_lookaheads=[1]):
    """ features and targets function: 
        features: - log returns with various lookbacks.
        targets: - log return
    """
    results = pd.DataFrame(index=df.index)
    feature_cols = []
    target_cols = []
    for lb in return_lookbacks:
        col = 'lret-' + str(lb)
        results[col] = logreturn(df['px'], df['px'].shift(lb))
        feature_cols.append(col)

        # add target feature to predict
    for la in target_lookaheads:
        col = 'target-' + str(la)
        results[col] = logreturn(df['px'].shift(-la), df['px'])
        target_cols.append(col)

    results = results.dropna()  # so that features and targets are all complete, and have aligned samples
    return results[feature_cols], results[target_cols]


def FT_ma_ewma_logreturns_vs_abs_logreturn(df, ma_windows=[10], ewma_halflifes=[10], lret_lookbacks=[]):
    """ features and targets function: 
        features: 
        - moving average of abs log return with various lookbacks.
        - ewma of abs log return with various lookbacks
        targets:
        - abs log return
    """
    results = pd.DataFrame(index=df.index)
    feature_cols = []
    target_cols = []

    lret = logreturn(df['px'], df['px'].shift(1))
    vol = lret.abs()
    results['vol'] = vol
    feature_cols.append('vol')
    future_vol = vol.shift(-1)
    for ma_win in ma_windows:
        col = 'ma-' + str(ma_win)
        ma = vol.rolling(ma_win).mean()
        results[col] = ma
        feature_cols.append(col)

    for ewma_hl in ewma_halflifes:
        col = 'ewma-' + str(ewma_hl)
        ewma = vol.ewm(halflife=ewma_hl).mean()
        results[col] = ewma
        feature_cols.append(col)

    for lb in lret_lookbacks:
        col = 'lret-' + str(lb)
        lret_lb = lret.shift(lb)
        results[col] = lret_lb
        feature_cols.append(col)

        # add target feature to predict
    results['target-1'] = future_vol
    target_cols.append('target-1')

    results = results.dropna()  # so that features and targets are all complete, and have aligned samples
    return results[feature_cols], results[target_cols]


def lret(px): return logreturn(px, px.shift(1))


def FT2_ma_ewma_logreturns_vs_abs_logreturn(ma_windows=[10], ewma_halflifes=[10], lret_lookbacks=[]):
    """ features and targets function:
        features:
        - moving average of abs log return with various lookbacks.
        - ewma of abs log return with various lookbacks
        - log return
        targets:
        - abs log return
    """
    features = {}
    targets = {}

    def vol(px):
        lret(px).abs()

    features['vol'] = vol
    targets['target-1'] = lambda x: vol(x).shift(-1)

    for ma_win in ma_windows:
        features['ma-' + str(ma_win)] = lambda x: vol(x).rolling(ma_win).mean()

    for ewma_hl in ewma_halflifes:
        features['ewma-' + str(ewma_hl)] = lambda x: vol(x).ewm(halflife=ewma_hl).mean()

    for lb in lret_lookbacks:
        features['lret-' + str(lb)] = lambda x: lret(x).shift(lb)

    return features, targets


def create_features_and_targets(series, feature_defs, target_defs):
    results = pd.DataFrame(index=series.index)
    for col, func in feature_defs.items():
        results[col] = func(series)

    for col, func in target_defs.items():
        results[col] = func(series)

    results = results.dropna()  # so that features and targets are all complete, and have aligned samples
    return results[feature_defs.keys()], results[target_defs.keys()]


def load_ds1():
    features_and_targets = partial(FT_logreturn_vs_logreturn, return_lookbacks=np.arange(40) + 1, target_lookaheads=[1])
    return load_stock_datasets(features_and_targets)


def load_ds2():
    features_and_targets = partial(FT_ma_ewma_logreturns_vs_abs_logreturn, ma_windows=np.arange(40) + 1,
                                   ewma_halflifes=np.arange(40) + 1)
    return load_stock_datasets(features_and_targets)


def load_ds2_rand(syms_to_std_map={'S1': 0.03}):
    features_and_targets = partial(FT_ma_ewma_logreturns_vs_abs_logreturn, ma_windows=np.arange(40) + 1,
                                   ewma_halflifes=np.arange(40) + 1)
    return create_random_datasets(features_and_targets, syms_to_std_map=syms_to_std_map)


def load_ds3():
    features_and_targets = partial(
        create_features_and_targets,
        feature_defs={'vol': lambda x: lret(x)},
        target_defs={'target-1': lambda x: lret(x).abs().shift(-1)})
    return create_random_datasets(features_and_targets)

def preview(*series, **kwargs):
    n = kwargs.get('n',4)
    return pd.concat([ser.head(n) for ser in series], axis=1)


def concatenate_datasets(data, name="combined"):
    def remove_index(x):
        return x.reset_index()[[c for c in x.columns if not 'index' == c]]

    X_train = pd.concat([remove_index(ds.X_train) for ds in data.values()])
    Y_train = pd.concat([remove_index(ds.Y_train) for ds in data.values()])
    X_dev = pd.concat([remove_index(ds.X_dev) for ds in data.values()])
    Y_dev = pd.concat([remove_index(ds.Y_dev) for ds in data.values()])
    X_test = pd.concat([remove_index(ds.X_test) for ds in data.values()])
    Y_test = pd.concat([remove_index(ds.Y_test) for ds in data.values()])

    ds = DataSet(name, X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
    return ds
