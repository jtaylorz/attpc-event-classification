"""
bulkDiscretize_p.py
===================

Bulk discretization of hdf5 formatted point cloud data.
"""
import sys
sys.path.insert(0, '../modules/')
import dataDiscretization as dd
import scipy as sp

#Whether or not we want to sum charge and add noise during discretization
CHARGE = True
NOISE = True

p_data = dd.bulkDiscretize('../data/tilt/p_40000_tilt.h5', 50, 50, 50, CHARGE, NOISE)
sp.sparse.save_npz('../data/tilt/50x50x50/pDisc_noise_40000_50x50x50_tilt.npz', p_data)

print (p_data.shape)
