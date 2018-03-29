"""
basicNN_sim2sim_pC.py
=====================

Testing a basic neural network on nuclear scattering data.
Trains on simulated tests on simulated.
Uses simulated events with > 30 points.
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

## 8 things to change total per run ##
#loading and splitting data
p_data = sp.sparse.load_npz('../data/tilt/20x20x20/pDisc_40000_20x20x20_tilt_largeEvts.npz')
C_data = sp.sparse.load_npz('../data/tilt/20x20x20/CDisc_40000_20x20x20_tilt_largeEvts.npz')

p_labels = np.zeros((p_data.shape[0],))
C_labels = np.ones((C_data.shape[0],))

full_data = sp.sparse.vstack([p_data, C_data], format='csr')
full_labels = np.hstack((p_labels, C_labels))
print(full_data.shape)
print(full_labels.shape)

X_train, X_test, labels_train, labels_test = train_test_split(full_data, full_labels, test_size=validation_split, random_state=42)

#define model
model = Sequential()
model.add(Dense(128, input_dim=full_data.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model with a validation set split
history = model.fit(X_train.todense(), labels_train,
                    validation_data=(X_test.todense(), labels_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[metrics])

print(history.history.keys())

# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Single Layer NN Accuracy - Simulated p vs. Simulated C (> 30 points)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('../plots/results/tilt/basicNN_pC_largeEvts_acc.pdf')
# # summarize history for loss
# plt.figure(2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Single Layer NN Loss - p vs. C')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('../plots/results/tilt/basicNN_pC_loss.pdf')

textfile = open('../keras-results/NN/sim2sim/pC.txt', 'w')
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
