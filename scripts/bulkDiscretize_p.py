"""
Testing discretization modules
"""
import sys
sys.path.insert(0, '/home/taylor/Documents/independent-research/modules/')
import dataDiscretization as dd
import scipy as sp

p_data = dd.bulkDiscretize('/home/taylor/Documents/independent-research/data/p_40000.h5', 20, 20, 20)
sp.sparse.save_npz('/home/taylor/Documents/independent-research/data/pDisc_40000.npz', p_data)

print (p_data.shape)

#C_data = dd.bulkDiscretize('/home/taylor/Documents/independent-research/data/C_40000.h5', 20, 20, 20)
#sp.sparse.save_npz('/home/taylor/Documents/independent-research/data/CDisc_40000.npz', C_data)

#print (C_data.shape)
