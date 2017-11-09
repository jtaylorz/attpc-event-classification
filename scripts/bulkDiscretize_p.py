"""
Bulk discretization of h5 formatted point cloud data.
"""
import sys
sys.path.insert(0, '../modules/')
import dataDiscretization as dd
import scipy as sp

#Whether or not we want to sum charge and add noise during discretization
CHARGE = True
NOISE = True

p_data = dd.bulkDiscretize('../data/NO-tilt/p_40000.h5', 20, 20, 10, CHARGE, NOISE)
sp.sparse.save_npz('../data/NO-tilt/20x20x10/pDisc_40000_20x20x10.npz', p_data)

print (p_data.shape)
