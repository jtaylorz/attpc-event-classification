"""
Testing discretization modules
"""
import sys
sys.path.insert(0, '/home/taylor/Documents/independent-research/modules/')
import dataDiscretization as dd
import scipy as sp

C_data = dd.bulkDiscretize('/home/taylor/Documents/independent-research/data/C_40000.h5', 20, 20, 20)
sp.sparse.save_npz('/home/taylor/Documents/independent-research/data/CDisc_40000.npz', C_data)

print (C_data.shape)
