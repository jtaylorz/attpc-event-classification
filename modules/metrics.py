"""
metrics.py
==========

Module defining metrics Callback function to save confusion matrices during
network training. Pass an instantiation of the Metrics() class to the callback
parameter of model.fit().

Inspiration from:
https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2

To be pasted into scripts (for recording and storing confusion matrices)

# import module
import sys
sys.path.insert(0, '../modules/')
import metrics
metrics = metrics.Metrics()

# write results to text file
textfile = open('../keras-results/CNN/sim2sim/pC.txt', 'w')
textfile.write('acc \n')
textfile.write(str(history.history['acc']))
textfile.write('\n')
textfile.write('\nval_acc \n')
textfile.write(str(history.history['val_acc']))
textfile.write('\n')
textfile.write('\nconfusion matrices \n')
for cm in metrics.val_cms:
    textfile.write(str(cm))
    textfile.write('\n')
"""
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class BinaryMetrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_cms = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_cm = confusion_matrix(val_targ, val_predict)
        self.val_cms.append(_val_cm)
        print('\n')
        print(_val_cm)
        return

class MulticlassMetrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_cms = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_cm = confusion_matrix(val_targ.argmax(axis=1), val_predict.argmax(axis=1))
        self.val_cms.append(_val_cm)
        # print('\n')
        # print(_val_cm)
        return

class MetricsOther(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        return
