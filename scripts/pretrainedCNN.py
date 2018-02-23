"""
pretrainedCNN.py
================

Testing pretrained convolutional neural network solution to event classification
problem. Model uses a VGG16 architecture pretrained on the ImageNet database to
learn basic and universal image rcognition features such as edge detection and
etc. A small top model is then trained on top of the VGG16 network to classify
our data.

Inputs are 128x128 pixel plots of events.
"""
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
#validation_split = 0.0

#load images from hdf5 files
hdf5_path = '../cnn-plots/hdf5s/'

sim_p_file = h5py.File(hdf5_path + 'sim_p.h5', 'r')
sim_C_file = h5py.File(hdf5_path + 'sim_C.h5', 'r')
sim_junk_file = h5py.File(hdf5_path + 'sim_junk.h5', 'r')

real_p_file = h5py.File(hdf5_path + 'real_p.h5', 'r')
real_C_file = h5py.File(hdf5_path + 'real_C.h5', 'r')
real_junk_file = h5py.File(hdf5_path + 'real_junk.h5', 'r')

sim_p = sim_p_file['img']
sim_C = sim_C_file['img']
sim_junk = sim_junk_file['img']

real_p = real_p_file['img']
real_C = real_C_file['img']
real_junk = real_junk_file['img']

print(sim_p.shape)
print(sim_C.shape)
print(sim_junk.shape)
print(real_p.shape)
print(real_C.shape)
print(real_junk.shape)
