"""
Script to generate 128x128 pixel images of simulated data. These images
will be the dataset used for a convolutional neural net test to classify protons,
carbon, and junk.
"""
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

import pytpc
import sys
sys.path.insert(0, '../modules/')
import dataDiscretization as dd

data_path = '../data/tilt/'
plot_path = '../cnn-plots/'

#proton events
with pytpc.HDFDataFile(data_path + 'p_40000_tilt.h5', 'r') as p_sim:
    n_evts = len(p_sim)
    evt_id = 0

    while (evt_id < n_evts):
        curEvt = p_sim[evt_id]
        curxyz = curEvt.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False, baseline_correction=False, cg_times=False)
        noise_xyz = dd.addNoise(curxyz)

        plt.figure(figsize=(1.28, 1.28), dpi=100)
        plt.plot(noise_xyz[:,2], noise_xyz[:,1],'.', markersize=2.0, c='black')
        plt.xlim(0.0, 1250.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.savefig(plot_path + 'sim_p/psim_evt_' + '{0:05d}'.format(evt_id) + '.png')
        plt.close()

        print("Plotted/saved simulated proton event " + str(evt_id))
        evt_id += 1

#carbon events
with pytpc.HDFDataFile(data_path + 'C_40000_tilt.h5', 'r') as C_sim:
    n_evts = len(p_sim)
    evt_id = 0

    while (evt_id < n_evts):
        curEvt = C_sim[evt_id]
        curxyz = curEvt.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False, baseline_correction=False, cg_times=False)
        noise_xyz = dd.addNoise(curxyz)

        plt.figure(figsize=(1.28, 1.28), dpi=100)
        plt.plot(noise_xyz[:,2], noise_xyz[:,1],'.', markersize=2.0, c='black')
        plt.xlim(0.0, 1250.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.savefig(plot_path + 'sim_C/Csim_evt_' + '{0:05d}'.format(evt_id) + '.png')
        plt.close()

        print("Plotted/saved simulated carbon event " + str(evt_id))
        evt_id += 1

# #junk events
# n_junk_evts = 40000
# evt_id = 0
#
# while (evt_id < n_junk_evts):
#     empty_evt = np.empty([1,4])
#     noise_evt = dd.addNoise(empty_evt)
#
#     plt.figure(figsize=(1.28, 1.28), dpi=100)
#     plt.plot(noise_evt[:,2], noise_evt[:,1],'.', markersize=2.0, c='black')
#     plt.xlim(0.0, 1250.0)
#     plt.ylim((-275.0, 275.0))
#     plt.axis('off')
#     plt.savefig(plot_path + 'sim_junk/junksim_evt_' + '{0:05d}'.format(evt_id) + '.png')
#     plt.close()
#
#     print("Plotted/saved simulated noise event " + str(evt_id))
#     evt_id += 1
