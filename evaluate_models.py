

from __future__ import division, print_function
import numpy as np
import utils
from model01 import MLPModel01
from metrics import performance_report
import datasets



n_categories = 2  # implicit in prepare_data (maybe parameterise)
lookahead = 1
window = 60
sym = 'USDJPY'

# In[21]:

X_train, Y_train, prices_train,_ = datasets.load(datasets.filename('DS2', lookahead, window, sym, 2009))

X_dev, Y_dev, prices_dev = datasets.load(datasets.filename('DS2', lookahead, window, sym, 2010))
# sample 50k records from 2010 as dev set
dev_idx = np.random.choice(len(X_dev), 50000, replace=False)
X_dev, Y_dev, prices_dev,_ = X_dev.ix[dev_idx], Y_dev.ix[dev_idx], prices_dev.ix[dev_idx]

X_test, Y_test, prices_test,_ = datasets.load(datasets.filename('DS2', lookahead, window, sym, 2011))

# In[23]:

print("train", X_train.shape)
print("dev", X_dev.shape)
print("test", X_test.shape)
n_features = X_train.shape[1]
print("n_features:", n_features)


layer_widths = [100, 100, 100]
dropout = 0.5

model = MLPModel01(lookahead, n_features, n_categories, layer_widths, dropout)
print(model.summary())

max_epochs = 200
hist = model.fit(
    X_train.as_matrix(), Y_train,
    validation_data=(X_dev.as_matrix(), Y_dev),
    max_epochs=max_epochs,
    es_patience=200,
    batch_size=1024 * 50)

# In[ ]:

# plt.plot(model.progress_callback.train_losses, label='train_loss')
# plt.plot(model.progress_callback.validation_losses, label='validation_loss')
# plt.plot(model.progress_callback.train_f1s, label='train_f1')
# plt.plot(model.progress_callback.validation_f1s, label='validation_f1')
# plt.plot(model.progress_callback.train_accuracies, label='train_accuracy')
# plt.plot(model.progress_callback.validation_accuracies, label='validation_accuracy')
# plt.legend();

# In[ ]:

Y_train_pred = model.predict(X_train.as_matrix(), batch_size=1024)
Y_train_pred_class = utils.prediction_to_category2(Y_train_pred)
Y_test_pred = model.predict(X_test.as_matrix(), batch_size=1024)
Y_test_pred_class = utils.prediction_to_category2(Y_test_pred)

# In[ ]:

# plt.figure(figsize=(8, 4))
# plt.hist(Y_train_pred, alpha=0.5, bins=30, normed=True, label='train')
# plt.hist(Y_test_pred, alpha=0.5, bins=30, normed=True, label='test')
# plt.hist(Y_train_pred_class, alpha=0.3, normed=True)
# plt.hist(Y_test_pred_class, alpha=0.3, normed=True)
# plt.legend()

# In[ ]:

performance_report("train", prices_train, lookahead, Y_train, Y_train_pred_class)
performance_report("test", prices_test, lookahead, Y_test, Y_test_pred_class)

#train_curve = precision_recall_curve(Y_train, Y_train_pred)
#test_curve = precision_recall_curve(Y_test, Y_test_pred)

# plt.plot(train_curve[0], train_curve[1], label='train')
# plt.plot(test_curve[0], test_curve[1], label='test')
# plt.legend()
# plt.show()
# sns.heatmap(confusion_matrix(Y_test, Y_test_pred_class))

