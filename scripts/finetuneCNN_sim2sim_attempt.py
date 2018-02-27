"""
finetuneCNN_sim2sim_attempt.py
=================================

Testing pretrained/finetunable convolutional neural network solution to event classification
problem. Model uses a VGG16 architecture pretrained on the ImageNet database to
learn basic and universal image rcognition features such as edge detection and
etc. A small top model is then trained on top of the VGG16 network to classify
our data.

Inputs are 128x128 pixel plots of events.
Tabling this (2/26/18) for non fine-tuning approach.
"""
import os
import numpy as np
import h5py

from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

seed = 7
np.random.seed(seed)
img_width, img_height = 128, 128
input_shape = (img_width, img_height, 3)

batch_size = 16
epochs = 25
validation_split = 0.25

#paths
hdf5_path = '../cnn-plots/hdf5s/'
bottleneck_features_path = '../models/bottleneck_features_sim2sim.npy'
top_model_weights_path = '../models/top_model_trained_sim2sim.h5'

#load images from hdf5 files
sim_p_file = h5py.File(hdf5_path + 'sim_p.h5', 'r')
sim_C_file = h5py.File(hdf5_path + 'sim_C.h5', 'r')
sim_junk_file = h5py.File(hdf5_path + 'sim_junk.h5', 'r')

sim_p = sim_p_file['img']
sim_C = sim_C_file['img']
sim_junk = sim_junk_file['img']

#labels
sim_p_labels = np.zeros((sim_p.shape[0],))
sim_C_labels = np.ones((sim_C.shape[0],))
sim_junk_labels = np.ones((sim_junk.shape[0],))

sim_X = np.vstack((np.array(sim_p), np.array(sim_C), np.array(sim_junk)))
sim_labels = np.hstack((sim_p_labels, sim_C_labels, sim_junk_labels))


def save_bottleneck_features():

    #build VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    print("Calculating pre-trained weights...")
    bottleneck_features_train  = model.predict(sim_X)
    np.save(open(bottleneck_features_train, 'wb'), bottleneck_features_train)


def train_top_model():
    train_data = np.load(open(bottleneck_features_path, 'rb'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(labels.shape[1], activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print("Training top model on training data")
    model.fit(train_data, labels_train,
              shuffle='batch',
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split)

    model.save_weights(top_model_weights_path)


if not os.path.isfile(bottleneck_features_path):
    save_bottleneck_features()
else:
    print("Pre-trained weights already calculated and stored")

if not os.path.isfile(top_model_weights_path):
    train_top_model()
else:
    print("Top model already trained and stored")
