"""

"""
from __future__ import division, print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

from keras.utils import np_utils
import pandas as pd
import numpy as np
import time


class MLPModel01:
    def __init__(self, lookahead, n_features, n_categories, layer_widths, dropout=0.0):
        self.lookahead = lookahead
        self.n_features = n_features
        self.n_categories = n_categories

        self.model = build_model(
            n_features=self.n_features,
            n_categories=self.n_categories,
            layer_widths=layer_widths,
            dropout=dropout
        )

    def summary(self):
        return self.model.summary()


def build_model(n_features, n_categories, layer_widths, dropout):

    start = time.time()
    model = Sequential()

    for i, w in enumerate(layer_widths):
        if i == 0:
            model.add(Dense(w, input_dim=n_features, activation='relu', init='he_normal'))
        else:
            model.add(Dense(w, activation='relu', init='he_normal'))

        if dropout > 0:
            model.add(Dropout(dropout))

    # final output layer
    if n_categories > 2:
        model.add(Dense(output_dim=n_categories, activation='softmax'))
        loss_function = 'categorical-cross-entropy'
    else:
        model.add(Dense(output_dim=1, activation='sigmoid'))
        loss_function = 'binary_crossentropy'

    model.compile(loss=loss_function, optimizer='adam')

    print('compilation took: {:.1f} seconds'.format(time.time()-start))
    return model


if __name__ == '__main__':
    print( MLPModel01(1, 20, 2, [50,100], 0).summary())

