import pandas as pd
import os, re, collections
from env import FX_DATASET_DIR

from keras.utils.np_utils import to_categorical


def hlc_to_price(df):
    return 0.5 * (df['close'] + 0.5 * df['high'] + 0.5 * df['low'])


def future_return(px, lookahead):
    return (px.shift(-lookahead) / px) - 1


def categoriser2(fut_return):
    return 1 * (fut_return > 0)


def fx_1minute_bar_catalog():
    res = collections.defaultdict(list)
    files = os.listdir(FX_DATASET_DIR)
    for f in files:
        m = re.search('DAT_ASCII_(\w+)_M1_(\d+).csv', f)
        if m:
            sym, date = m.groups()
            res[sym].append(date)
    return res


def load_1minute_fx_bars(sym, date):
    filename = os.path.join(FX_DATASET_DIR, 'DAT_ASCII_{sym}_M1_{date}.csv'.format(sym=sym, date=date))
    df = pd.read_csv(filename, header=None, sep=';',
                     names=['ts', 'open', 'high', 'low', 'close', 'volume']
                     )
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True, verify_integrity=True)
    # volume always empty in this dataset
    del df['volume']
    return df
