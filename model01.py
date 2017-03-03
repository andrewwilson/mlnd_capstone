"""

"""
from __future__ import division, print_function

import datetime
import os
import time

import keras
import numpy as np
from keras.layers.core import Dense, Dropout
from keras.models import Sequential

import env


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
        self.model_id = "MLPModel01_{:%Y%m%d_%H%M}_LA{}_F{}_C{}_L{}_DO{}".format(
            datetime.datetime.now(),
            lookahead, n_features, n_categories, "_".join([str(l) for l in layer_widths]), dropout
        )
        print("Model id: ", self.model_id)
        self.progress_callback = ProgressCallback(self.model_id)

    def summary(self):
        return self.model.summary()

    def fit(self, X, Y, validation_data=None, max_epochs=10, batch_size=1024, save_model_epochs=5, es_min_delta=0.001,
            es_patience=10):
        self.progress_callback.save_model_epochs = save_model_epochs

        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=es_min_delta,
                                                                patience=es_patience, verbose=2, mode='min')

        learning_rate_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                                                                   verbose=1, mode='min',
                                                                   epsilon=0.0001, cooldown=2, min_lr=5e-5)
        if not validation_data:
            validation_split = 0.1
        else:
            validation_split = 0
        self.model.fit(X, Y,
                       verbose=0, validation_split=validation_split, batch_size=batch_size,
                       validation_data=validation_data,
                       nb_epoch=max_epochs,
                       callbacks=[
                           self.progress_callback,
                           early_stopping_callback,
                           learning_rate_callback
                       ])

        return self.progress_callback

    def predict(self, X, batch_size=1024):
        return self.model.predict(X, batch_size)


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

    model.compile(loss=loss_function, optimizer='adam', metrics=['fmeasure', 'accuracy'])

    print('compilation took: {:.1f} seconds'.format(time.time() - start))
    return model


class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, run_id, epoch_count=0,
                 train_losses=[],
                 validation_losses=[],
                 train_accuracies=[],
                 validation_accuracies=[],
                 train_f1s=[],
                 validation_f1s=[],
                 save_model_epochs=5):
        self.run_id = run_id
        self.train_losses = train_losses
        self.validation_losses = validation_losses
        self.train_accuracies = train_accuracies
        self.validation_accuracies = validation_accuracies
        self.train_f1s = train_f1s
        self.validation_f1s = validation_f1s
        self.epoch_count = epoch_count
        self.last_model_file = None
        self.save_model_epochs = save_model_epochs

    def on_epoch_end(self, epoch, logs):
        #print(logs)
        train_loss = logs['loss']
        validation_loss = logs['val_loss']
        train_f1 = logs['fmeasure']
        validation_f1 = logs['val_fmeasure']
        train_accuracy = logs['acc']
        validation_accuracy = logs['val_acc']
        self.train_losses.append(train_loss)
        self.validation_losses.append(validation_loss)
        self.train_accuracies.append(train_accuracy)
        self.validation_accuracies.append(validation_accuracy)
        self.train_f1s.append(train_f1)
        self.validation_f1s.append(validation_f1)

        print("Epoch {}, train_loss: {:.4f} validation_loss {:.4f}  validation accuracy {:.4f}  validation f1 {:.4f}"
              .format(epoch, train_loss, validation_loss, validation_accuracy, validation_f1))

        self.epoch_count += 1
        if self.epoch_count % self.save_model_epochs == 0:
            self.last_model_file = self.model_filename(self.run_id, self.epoch_count, validation_loss)
            self.model.save(self.last_model_file)
            self.save()
            print("saved to ", self.last_model_file)

    def save(self):
        np.savez(
            self.exp_filename(self.run_id),
            run_id=self.run_id,
            epoch_count=self.epoch_count,
            tl=np.array(self.train_losses),
            dl=np.array(self.validation_losses),
            last_model_file=self.last_model_file)
        # todo: load and save accuracies and f1s??

    @staticmethod
    def exp_filename(run_id):
        """ the filename to use for the experiment summary file for the given run_id.
        """
        return os.path.join(env.SAVES_DIR, run_id + ".npz")

    @staticmethod
    def model_filename(run_id, epoch, loss):
        """ the filename to use for the model state file for the given run_id
        """
        return os.path.join(env.SAVES_DIR, '{}-{:03d}-{:.4f}.hdf5'.format(run_id, epoch, loss))

    @classmethod
    def load(cls, run_id):
        """
        Loads an experiment summary and latest saved model state from file
        :param run_id:
        :return:
        """
        fname = cls.exp_filename(run_id)
        print("loading from file:", fname)
        npz = np.load(fname)
        assert run_id == str(npz['run_id'])
        epoch_count = int(npz['epoch_count'])
        tl = list(npz['tl'])
        dl = list(npz['dl'])
        last_model_file = str(npz['last_model_file'])
        experiment = ProgressCallback(run_id, epoch_count, tl, dl)
        experiment.last_model_file = last_model_file
        return experiment

    @classmethod
    def create_or_load(cls, run_id):
        fname = cls.exp_filename(run_id)
        if os.path.exists(fname):
            prog = cls.load(run_id)
            print("loaded progress: ", prog.run_id, "training epochs:", prog.epoch_count)
            return prog
        else:
            print("creating new progress for run id:", run_id)
            return ProgressCallback(run_id)


if __name__ == '__main__':
    import preprocess
    import utils

    lookahead = 1
    window = 10
    X_train, Y_train, prices_train = preprocess.prepare_data(utils.load_1minute_fx_bars("USDJPY", 2009)[:10000],
                                                             lookahead=lookahead, window=window)

    X_dev, Y_dev, prices_dev = preprocess.prepare_data(utils.load_1minute_fx_bars("USDJPY", 2009)[10000:20000],
                                                       lookahead=lookahead, window=window)

    model = MLPModel01(lookahead=lookahead, n_features=X_train.shape[1], n_categories=2, layer_widths=[1], dropout=0)
    print(model.summary())

    prog = model.fit(X_train.as_matrix(), Y_train, validation_data=(X_dev.as_matrix(), Y_dev), max_epochs=100,
                     es_min_delta=0.0001, es_patience=10)
    print(prog)
