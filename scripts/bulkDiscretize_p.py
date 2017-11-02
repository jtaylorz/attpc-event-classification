"""
Testing discretization modules
"""
import sys
sys.path.insert(0, '/home/taylor/Documents/independent-research/modules/')
import dataDiscretization as dd
import scipy as sp

#Whether or not we want to sum charge during discretization
CHARGE = False

p_data = dd.bulkDiscretize('/home/taylor/Documents/independent-research/data/20x20x20/p_40000.h5', 20, 20, 20, CHARGE)
sp.sparse.save_npz('/home/taylor/Documents/independent-research/data/20x20x20/pDisc_40000_nocharge.npz', p_data)

print (p_data.shape)
