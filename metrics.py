from __future__ import division, print_function
from sklearn.metrics import f1_score, classification_report


def performance_report(name, price_series, lookahead, true_class, predicted_class):
    print("------------------------------------------------------")
    print(name)
    print("f1-score: {:.3f}".format(
            f1_score(true_class, predicted_class, average='weighted')
        ))



    #print("accuracy: {:.3f}".format(accuracy_score(true_class, predicted_class)))
    print(classification_report(true_class, predicted_class))

