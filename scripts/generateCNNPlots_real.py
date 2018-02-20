"""
generateCNNPlots_real.py
========================

Script to generate 128x128 pixel images of real data. These images
will be the dataset used for a convolutional neural net test to classify protons,
carbon, and junk.
"""
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import pandas as pd

import pytpc
import sys
sys.path.insert(0, '../modules/')
import dataDiscretization as dd

#load data
data_path = '../data/real/'
disc_path = data_path + '50x50x50/'
runs = ['0130', '0210']

plot_path = '../cnn-plots/'

for run in runs:
    data = pytpc.HDFDataFile(data_path + "run_" + run + ".h5", 'r')
    labels = pd.read_csv(data_path + "run_" + run + "_labels.csv", sep=',')
    print("Successfully loaded data and labels for run " + str(run) + ".")

    #plot and save proton events
    p_indices = labels.loc[(labels['label'] == 'p')]['evt_id'].values

    for evt_id in p_indices:
        curEvt = data[evt_id]
        curxyz = curEvt.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False, baseline_correction=False, cg_times=False)

        plt.figure(figsize=(1.28, 1.28), dpi=100)
        plt.plot(curxyz[:,2], curxyz[:,1],'.', markersize=2.0, c='black')
        plt.xlim(0.0, 1250.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.savefig(plot_path + 'real_p/run_' + run + '_evt_' + '{0:05d}'.format(evt_id) + '.png')
        plt.close()

        print("Plotted/saved run_" + run + " proton event " + str(evt_id))

    #plot and save carbon events
    C_indices = labels.loc[(labels['label'] == 'c')]['evt_id'].values

    for evt_id in C_indices:
        curEvt = data[evt_id]
        curxyz = curEvt.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False, baseline_correction=False, cg_times=False)

        plt.figure(figsize=(1.28, 1.28), dpi=100)
        plt.plot(curxyz[:,2], curxyz[:,1],'.', markersize=2.0, c='black')
        plt.xlim(0.0, 1250.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.savefig(plot_path + 'real_C/run_' + run + '_evt_' + '{0:05d}'.format(evt_id) + '.png')
        plt.close()

        print("Plotted/saved run_" + run + " carbon event " + str(evt_id))

    #plot and save proton events
    junk_indices = labels.loc[(labels['label'] == 'j')]['evt_id'].values

    for evt_id in junk_indices:
        curEvt = data[evt_id]
        curxyz = curEvt.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False, baseline_correction=False, cg_times=False)

        plt.figure(figsize=(1.28, 1.28), dpi=100)
        plt.plot(curxyz[:,2], curxyz[:,1],'.', markersize=2.0, c='black')
        plt.xlim(0.0, 1250.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.savefig(plot_path + 'real_junk/run_' + run + '_evt_' + '{0:05d}'.format(evt_id) + '.png')
        plt.close()

        print("Plotted/saved run_" + run + " junk event " + str(evt_id))
