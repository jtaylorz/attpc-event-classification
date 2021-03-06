"""
basicNN_sim2sim_multic.py
==========================

Testing a basic neural network on nuclear scattering data.
Trains on simulated tests on simulated.
"""
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import scipy as sp

#fix random seed
np.random.seed(7)
epochs = 100
validation_split = 0.25
batch_size = 10

import sys
sys.path.insert(0, '../modules/')
import metrics
metrics = metrics.MulticlassMetrics()

## 8 things to change total per run ##
#loading and splitting data
p_data = sp.sparse.load_npz('../data/tilt/20x20x20/pDisc_40000_20x20x20_tilt_largeEvts.npz')
C_data = sp.sparse.load_npz('../data/tilt/20x20x20/CDisc_40000_20x20x20_tilt_largeEvts.npz')
noise_data = sp.sparse.load_npz('../data/tilt/20x20x20/noiseDisc_40000_20x20x20.npz')

p_labels = np.zeros((p_data.shape[0],))
C_labels = np.ones((C_data.shape[0],))
noise_labels = np.full((noise_data.shape[0],), 2)

# full_data = sp.sparse.vstack([p_data, C_data], format='csr')
# full_labels = np.hstack((p_labels, C_labels))
full_data = sp.sparse.vstack([p_data, C_data, noise_data], format='csr')
full_labels_categorical = np.hstack((p_labels, C_labels, noise_labels))
full_labels = np_utils.to_categorical(full_labels_categorical)

print(full_data.shape)
print(full_labels_categorical.shape)
print(full_labels.shape)

X_train, X_test, labels_train, labels_test = train_test_split(full_data, full_labels, test_size=validation_split, random_state=42)

#define model
model = Sequential()
model.add(Dense(128, input_dim=full_data.shape[1], activation='relu'))
model.add(Dense(full_labels.shape[1], activation='softmax'))

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
plt.title('Single Layer NN Accuracy - p vs. C + junk')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('../plots/results/tilt/basicNN_sim_pCjunk_acc.pdf')
# # summarize history for loss
# plt.figure(2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Single Layer NN Loss - p vs. C + junk')
# plt.ylabel('loss')25
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], l25oc='upper left')
# plt.savefig('../plots/results/tilt/basicNN_sim_pCjunk_loss.pdf')

textfile = open('../keras-results/NN/sim2sim/multic.txt', 'w')
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
