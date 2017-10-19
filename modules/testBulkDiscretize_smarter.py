"""
Testing discretization modules
"""
import dataDiscretization_smarter as dds
import scipy as sp

p_data = dds.bulkDiscretize('/home/taylor/Documents/independent-research/data/p_40000.h5', 20, 20, 20)
#sp.sparse.save_npz('/home/taylor/Documents/independent-research/data/test.npz', p_data)

print (p_data.shape)

#C_data = dds.bulkDiscretize('/home/taylor/Documents/independent-research/data/C_40000.h5', 20, 20, 20)
#sp.sparse.save_npz('/home/taylor/Documents/independent-research/data/test.npz', p_data)

#print (C_data.shape)
