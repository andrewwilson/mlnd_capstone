from __future__ import division, print_function
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import utils


def performance_report(name, price_series, lookahead, true_class, prediction,
                       f1_detail=False,
                       cum_return_plot=False,
                       histogram=False,
                       heatmap=False,
                       savefig=None,
                       no_print=False
                       ):
    """
    Generates a performance report
    :param name: name of the model to report
    :param price_series: the raw price series
    :param lookahead: the number of periods look-ahead to consider for the future-return prediction
    :param true_class: the true future return direction
    :param prediction: the real valued prediction of the model
    :param f1_detail: flag - should more details of f1 score be reported 
    :param cum_return_plot: flag - should cumulative future return chart be drawn
    :param histogram: flag - should a histogram of the predictions be drawn
    :param heatmap: should a heatmap of actual versus predicted directions be drawn.
    :param savefig: if specified, indicates the filename under which to save the last drawn figure
    :param no_print: flag - suppress result printing
    :return dict of the metrics
    
    """
    predicted_class = utils.prediction_to_category2(prediction)

    fut_returns = utils.future_return(price_series, lookahead).fillna(0)
    pred_fut_return_demeaned = predicted_future_return(fut_returns, predicted_class, demean=True)
    pred_fut_return_non_demeaned = predicted_future_return(fut_returns, predicted_class, demean=False)

    metrics = {
        'name': name,
        'f1_score': f1_score(true_class, predicted_class, average='weighted'),
        'mean_fut_return': pred_fut_return_demeaned.mean() * 1e4,
        'mean_fut_return_non_dm': pred_fut_return_non_demeaned.mean() * 1e4,
        'ann_fut_return': annualized_mean_future_return(pred_fut_return_demeaned, lookahead_minutes=lookahead),
        'ann_fut_return_non_dm': annualized_mean_future_return(pred_fut_return_non_demeaned, lookahead_minutes=lookahead)
    }

    if not no_print:
        print("{name}: f1-score: {f1_score:.3f}, mean future return: {mean_fut_return:.3f} bps,"
            " ({mean_fut_return_non_dm:.3f} bps),"
              " annualized future return {ann_fut_return:.3f} ({ann_fut_return_non_dm:.3f})".format(**metrics))

    if f1_detail:
        print(classification_report(true_class, predicted_class))

    if cum_return_plot:
        fut_return_plot(name, pred_fut_return_demeaned, savefig)

    if histogram:
        histo(name, prediction, savefig)

    if heatmap:
        show_heatmap(name, true_class, predicted_class, savefig)

    return metrics


def annualized_mean_future_return(pred_future_return, lookahead_minutes):
    """
    returns the annualized mean future return
    :param pred_future_return: 
    :param lookahead_minutes: 
    :return: 
    """
    one_minute_periods_per_year = 260*24*60
    num_periods_per_year = one_minute_periods_per_year/lookahead_minutes
    return ((pred_future_return.mean() + 1) ** (num_periods_per_year-1) ) -1


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


def fut_return_plot(name, pred_fut_return, savefig=None):
    import matplotlib.pyplot as plt
    (pred_fut_return + 1).cumprod().plot()
    plt.title(name)
    if savefig:
        plt.savefig(savefig)
    plt.show()


def histo(name, preds, savefig=None):
    import matplotlib.pyplot as plt

    plt.hist(preds, alpha=0.5, bins=30, normed=True, label=name)
    if savefig:
        plt.savefig(savefig)
    plt.show()


def show_heatmap(name, true_class, predicted_class, savefig=None):
    import seaborn as sns
    import matplotlib.pyplot as plt

    conf = confusion_matrix(true_class, predicted_class)
    print (conf)
    sns.heatmap(conf)
    if savefig:
        plt.savefig(savefig)

