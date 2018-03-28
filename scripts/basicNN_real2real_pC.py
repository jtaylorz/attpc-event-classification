"""
basicNN_real2real_pC.py
======================

Testing a basic neural network on nuclear scattering data.
Splits real labeled event dataset
Trains on real data and tests on real.
"""
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import scipy as sp

import sys
sys.path.insert(0, '../modules/')
import metrics
metrics = metrics.BinaryMetrics()

#fix random seed
np.random.seed(7)
epochs = 100
validation_split = 0.25
batch_size = 10

#loading real data
p_0130 = sp.sparse.load_npz('../data/real/20x20x20/run_0130_pDisc.npz')
C_0130 = sp.sparse.load_npz('../data/real/20x20x20/run_0130_CDisc.npz')
p_0210 = sp.sparse.load_npz('../data/real/20x20x20/run_0210_pDisc.npz')
C_0210 = sp.sparse.load_npz('../data/real/20x20x20/run_0210_CDisc.npz')

p_real = sp.sparse.vstack([p_0130, p_0210], format='csr')
C_real = sp.sparse.vstack([C_0130, C_0210], format='csr')

#creating labels
p_real_labels = np.zeros((p_real.shape[0],))
C_real_labels = np.ones((C_real.shape[0],))

#form proton/carbon sets
pC_real = sp.sparse.vstack([p_real, C_real], format='csr')
pC_real_labels = np.hstack((p_real_labels, C_real_labels))

#split data
(pC_real_train, pC_real_test,
pC_real_labels_train, pC_real_labels_test) = train_test_split(pC_real,
                                                            pC_real_labels,
                                                            test_size=0.25,
                                                            random_state=42)

#define model
model = Sequential()
model.add(Dense(128, input_dim=pC_real.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(pC_real_train.todense(), pC_real_labels_train,
                    validation_data=(pC_real_test.todense(), pC_real_labels_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[metrics])

print(history.history.keys())
# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Single Layer NN Accuracy - p vs. C - (> 30 points)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['simulated training data', 'real test data'], loc='upper left')
#plt.savefig('../plots/results/real/basicNN_sim2real_pC_largeEvts_acc.pdf')
# # summarize history for loss
# plt.figure(2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Single Layer NN Loss - p vs. C')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('../plots/results/tilt/basicNN_pC_loss.pdf')

textfile = open('../keras-results/NN/real2real/pC.txt', 'w')
textfile.write('acc \n')
textfile.write(str(history.history['acc']))
textfile.write('\n')
textfile.write('\nval_acc \n')
textfile.write(str(history.history['val_acc']))
textfile.write('\n')
textfile.write('\nloss \n')
textfile.write(str(history.history['loss']))
textfile.write('\n')
textfile.write('\nval_loss \n')
textfile.write(str(history.history['val_loss']))
textfile.write('\n')
textfile.write('\nconfusion matrices \n')
for cm in metrics.val_cms:
    textfile.write(str(cm))
    textfile.write('\n')

print("Maximum Validation Accuracy Reached: %.5f%%" % max(history.history['val_acc']))

#model_path = '../models/20x20x20/'

# # serialize model to YAML
# model_yaml = model.to_yaml()
# with open(model_path + "basicNN_NOISE.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
# # serialize weights to HDF5
# model.save_weights(model_path + "basicNN_NOISE.h5")
# print("Saved model to disk")
