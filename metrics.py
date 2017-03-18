from __future__ import division, print_function
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import utils


def performance_report(name, price_series, lookahead, true_class, prediction,
                       f1_detail=False,
                       cum_return_plot=False,
                       histogram=False,
                       heatmap=False
                       ):
    predicted_class = utils.prediction_to_category2(prediction)

    fut_returns = utils.future_return(price_series, lookahead).fillna(0)
    pred_fut_return = predicted_future_return(fut_returns, predicted_class, demean=True)


    print("{name}: f1-score: {f1:.3f}, mean future return: {mean_fut_return:.3f} bps".format(
        name=name,
        f1=f1_score(true_class, predicted_class, average='weighted'),
        mean_fut_return = pred_fut_return.mean() * 1e4
        ))

    if f1_detail:
        print(classification_report(true_class, predicted_class))

    if cum_return_plot:
        fut_return_plot(name, pred_fut_return)

    if histogram:
        histo(name, prediction)

    if heatmap:
        show_heatmap(name, true_class, predicted_class)


def predicted_future_return(fut_returns, predicted_class, demean=True):

    if demean:
        fut_returns = fut_returns - fut_returns.mean()

    predicted_class = predicted_class.ravel()
    l = len(fut_returns)
    mult = np.zeros(l)
    mult[predicted_class == 0] = -1
    mult[predicted_class == 1] = 1
    return fut_returns * mult


def aggregated_predicted_future_return(prices, prediction, agg_period, mode='sma', demean=True):

    predicted_class = utils.prediction_to_category2(prediction)
    fut_returns = utils.future_return(prices, agg_period).fillna(0)
    if demean:
        fut_returns = fut_returns - fut_returns.mean()

    predicted_class = predicted_class.ravel()

    if mode == 'sma':
        agg = pd.Series(predicted_class).rolling(window=agg_period)
    elif mode == 'ewm':
        agg = pd.Series(predicted_class).ewm(com=agg_period)
    else:
        raise ValueError("Unexpected mode: {}. Expected 'sma' or 'ewm'".format(mode))
    agg_predicted_class = agg.mean().values.astype(int)

    l = len(fut_returns)
    mult = np.zeros(l)
    mult[agg_predicted_class == 0] = -1
    mult[agg_predicted_class == 1] = 1
    return fut_returns * mult




def fut_return_plot(name, pred_fut_return):
    import matplotlib.pyplot as plt
    (pred_fut_return + 1).cumprod().plot()
    plt.title(name)
    plt.show()


def histo(name, preds):
    import matplotlib.pyplot as plt

    plt.hist(preds, alpha=0.5, bins=30, normed=True, label=name)
    plt.show()


def show_heatmap(name, true_class, predicted_class):
    import seaborn as sns

    conf = confusion_matrix(true_class, predicted_class)
    print (conf)
    sns.heatmap(conf)
