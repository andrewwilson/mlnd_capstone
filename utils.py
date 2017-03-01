import pandas as pd
import os, re, collections
from env import FX_DATASET_DIR


# def hlc_to_price(df):
#     return 0.5 * (df['close'] + 0.5 * df['high'] + 0.5 * df['low'])
#

def future_return(px, lookahead):
    """
    Computes the future return of a price series
    :param px: price series, as a pandas Series.
    :param lookahead: the number of periods into the future for which the returns should be calculated
    :return: future return series.
    """
    return (px.shift(-lookahead) / px) - 1


def categoriser2(return_series):
    """
    Categorises the given pandas return series into positive(=1) and negative(=0) categories.
    :param fut_return:  pandas Series or numpy array of price returns
    :return: pandas Series or numpy array of integer categories from (0,1)
    """
    return 1 * (return_series > 0)

def prediction_to_category2(prediction):
    """
    convert a series of real numbers between 0 and 1 to binary category in: (0,1)
    :param prediction: series of floats
    :return: series of int from (0,1)
    """
    return 1*prediction > 0.5


def fx_1minute_bar_catalog():
    """
    Convenience function to report the available downloaded FX 1minute bar datasets
    :return: dictionary of symbol versus list of years for which data has been downloaded.
    """
    res = collections.defaultdict(list)
    files = os.listdir(FX_DATASET_DIR)
    for f in files:
        m = re.search('DAT_ASCII_(\w+)_M1_(\d+).csv', f)
        if m:
            sym, date = m.groups()
            res[sym].append(date)
    return res


def load_1minute_fx_bars(sym, date):
    """
    Load dataframe of open, high, low, close FX price data, in "Generic Ascii CSV" format provided by histdata.com
    :param sym: the iso fx pair symbol, e.g. EURUSD
    :param date: the year, whose data should be loaded
    :return: pandas dataframe with columns: open, high, low, close,  with a datetime index.
    """
    filename = os.path.join(FX_DATASET_DIR, 'DAT_ASCII_{sym}_M1_{date}.csv'.format(sym=sym, date=date))
    df = pd.read_csv(filename, header=None, sep=';',
                     names=['ts', 'open', 'high', 'low', 'close', 'volume']
                     )
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True, verify_integrity=True)
    # volume always empty in this dataset
    del df['volume']
    return df
