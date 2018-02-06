"""
Bulk generation and discretization of h5 simulated noise events.
"""
import sys
sys.path.insert(0, '../modules/')
import dataDiscretization as dd
import scipy as sp

num_evts = 40000

#Whether or not we want to sum charge
CHARGE = True

noise_data = dd.createNoiseEvents(num_evts, 50, 50, 50, CHARGE)
sp.sparse.save_npz('../data/tilt/50x50x50/noiseDisc_40000_50x50x50.npz', noise_data)

print (noise_data.shape)
