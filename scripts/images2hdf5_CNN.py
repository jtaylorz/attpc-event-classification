"""
images2hdf5_CNN.py
==================

Script to save converted image data to separate HDF5 files for better use of
computational time when training networks.
"""
import numpy as np
import h5py
import glob
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image as pil_image

img_path = '../cnn-plots/'
hdf5_path = img_path + 'hdf5s/'

img_width, img_height = 128, 128

# #real protons
# real_p_filepaths = glob.glob(img_path + '/real_p/*.png')
# #1 channel for black & white images??
# real_p_inputshape = (len(real_p_filepaths), img_width, img_height, 3)
#
# real_p_hdf5 = h5py.File(hdf5_path + 'real_p.h5')
# real_p_hdf5.create_dataset("evt_id", (len(real_p_filepaths),), int)
# real_p_hdf5.create_dataset("img", real_p_inputshape, np.int8)
#
# for i in range(len(real_p_filepaths)):
#
#     if(i % 100 == 0):
#         print("Saved: {}/{} real proton events".format(i, len(real_p_filepaths)))
#
#     #parse out evt_id from image name
#     evt_id = int(real_p_filepaths[i][-9:-4])
#     #load plot image
#     img = load_img(real_p_filepaths[i], target_size=(img_width, img_height))
#     img_array = img_to_array(img) #Numpy array with shape (128, 128, 3)
#
#     real_p_hdf5['evt_id'][i, ...] = evt_id
#     real_p_hdf5['img'][i, ...] = img
#
# real_p_hdf5.close()
#
#
# #real carbons
# real_C_filepaths = glob.glob(img_path + '/real_C/*.png')
# real_C_inputshape = (len(real_C_filepaths), img_width, img_height, 3)
#
# real_C_hdf5 = h5py.File(hdf5_path + 'real_C.h5')
# real_C_hdf5.create_dataset("evt_id", (len(real_C_filepaths),), int)
# real_C_hdf5.create_dataset("img", real_C_inputshape, np.int8)
#
# for i in range(len(real_C_filepaths)):
#
#     if(i % 100 == 0):
#         print("Saved: {}/{} real carbon events".format(i, len(real_C_filepaths)))
#
#     evt_id = int(real_C_filepaths[i][-9:-4])
#     img = load_img(real_C_filepaths[i], target_size=(img_width, img_height))
#     img_array = img_to_array(img) #Numpy array with shape (128, 128, 3)
#
#     real_C_hdf5['evt_id'][i, ...] = evt_id
#     real_C_hdf5['img'][i, ...] = img
#
# real_C_hdf5.close()
#
#
# #real junk
# real_junk_filepaths = glob.glob(img_path + '/real_junk/*.png')
# real_junk_inputshape = (len(real_junk_filepaths), img_width, img_height, 3)
#
# real_junk_hdf5 = h5py.File(hdf5_path + 'real_junk.h5')
# real_junk_hdf5.create_dataset("evt_id", (len(real_junk_filepaths),), int)
# real_junk_hdf5.create_dataset("img", real_junk_inputshape, np.int8)
#
# for i in range(len(real_junk_filepaths)):
#
#     if(i % 100 == 0):
#         print("Saved: {}/{} real junk events".format(i, len(real_junk_filepaths)))
#
#     evt_id = int(real_junk_filepaths[i][-9:-4])
#     img = load_img(real_junk_filepaths[i], target_size=(img_width, img_height))
#     img_array = img_to_array(img) #Numpy array with shape (128, 128, 3)
#
#     real_junk_hdf5['evt_id'][i, ...] = evt_id
#     real_junk_hdf5['img'][i, ...] = img
#
# real_junk_hdf5.close()


# #simulated protons
# sim_p_filepaths = glob.glob(img_path + '/sim_p_largeEvts/*.png')
# sim_p_inputshape = (len(sim_p_filepaths), img_width, img_height, 3)
#
# sim_p_hdf5 = h5py.File(hdf5_path + 'sim_p_largeEvts.h5')
# sim_p_hdf5.create_dataset("evt_id", (len(sim_p_filepaths),), int)
# sim_p_hdf5.create_dataset("img", sim_p_inputshape, np.int8)
#
# for i in range(len(sim_p_filepaths)):
#
#     if(i % 100 == 0):
#         print("Saved: {}/{} simulated proton events".format(i, len(sim_p_filepaths)))
#
#     evt_id = int(sim_p_filepaths[i][-9:-4])
#     img = load_img(sim_p_filepaths[i], target_size=(img_width, img_height))
#     img_array = img_to_array(img) #Numpy array with shape (128, 128, 3)
#
#     sim_p_hdf5['evt_id'][i, ...] = evt_id
#     sim_p_hdf5['img'][i, ...] = img
#
# sim_p_hdf5.close()


#simulated carbons
sim_C_filepaths = glob.glob(img_path + '/sim_C_largeEvts/*.png')
sim_C_inputshape = (len(sim_C_filepaths), img_width, img_height, 3)

sim_C_hdf5 = h5py.File(hdf5_path + 'sim_C_largeEvts.h5')
sim_C_hdf5.create_dataset("evt_id", (len(sim_C_filepaths),), int)
sim_C_hdf5.create_dataset("img", sim_C_inputshape, np.int8)

for i in range(len(sim_C_filepaths)):

    if(i % 100 == 0):
        print("Saved: {}/{} simulated carbon events".format(i, len(sim_C_filepaths)))

    evt_id = int(sim_C_filepaths[i][-9:-4])
    img = load_img(sim_C_filepaths[i], target_size=(img_width, img_height))
    img_array = img_to_array(img) #Numpy array with shape (128, 128, 3)

    sim_C_hdf5['evt_id'][i, ...] = evt_id
    sim_C_hdf5['img'][i, ...] = img

sim_C_hdf5.close()


# #simulated noise
# sim_junk_filepaths = glob.glob(img_path + '/sim_junk/*.png')
# sim_junk_inputshape = (len(sim_junk_filepaths), img_width, img_height, 3)
#
# sim_junk_hdf5 = h5py.File(hdf5_path + 'sim_junk.h5')
# sim_junk_hdf5.create_dataset("evt_id", (len(sim_junk_filepaths),), int)
# sim_junk_hdf5.create_dataset("img", sim_junk_inputshape, np.int8)
#
# for i in range(len(sim_junk_filepaths)):
#
#     if(i % 100 == 0):
#         print("Saved: {}/{} simulated junk events".format(i, len(sim_junk_filepaths)))
#
#     evt_id = int(sim_junk_filepaths[i][-9:-4])
#     img = load_img(sim_junk_filepaths[i], target_size=(img_width, img_height))
#     img_array = img_to_array(img) #Numpy array with shape (128, 128, 3)
#
#     sim_junk_hdf5['evt_id'][i, ...] = evt_id
#     sim_junk_hdf5['img'][i, ...] = img
#
# sim_junk_hdf5.close()
