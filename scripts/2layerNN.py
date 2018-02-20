"""
2layerNN.py
===========

Testing a 2 layer neural network on nuclear scattering data.
"""
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import scipy as sp
import h5py

#fix random seed
np.random.seed(7)

#loading and splitting data
p_data = sp.sparse.load_npz('/home/taylor/Documents/independent-research/data/20x20x20/pDisc_40000_nocharge.npz')
C_data = sp.sparse.load_npz('/home/taylor/Documents/independent-research/data/20x20x20/CDisc_40000_nocharge.npz')
p_labels = np.zeros((p_data.shape[0],))
C_labels = np.ones((C_data.shape[0],))

full_data = sp.sparse.vstack([p_data, C_data], format='csr')
full_labels = np.hstack((p_labels, C_labels))
print(full_data.shape)
print(full_labels.shape)

#X_train, X_test, y_train, y_test = train_test_split(full_data, full_labels, test_size=0.25, random_state=0)

#define model
model = Sequential()
model.add(Dense(128, input_dim=full_data.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model with a validation set split
model.fit(full_data.todense(), full_labels, validation_split=0.25, epochs=50, batch_size=10)

#evaluate the model
scores = model.evaluate(full_data.todense(), full_labels, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


model_path = '/home/taylor/Documents/independent-research/networks/20x20x20/'

# # serialize model to YAML
# model_yaml = model.to_yaml()
# with open(model_path + "2layerNN.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
# # serialize weights to HDF5
# model.save_weights(model_path + "2layerNN.h5")
# print("Saved model to disk")
