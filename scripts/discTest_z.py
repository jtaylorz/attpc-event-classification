"""
Discretization script for discretizing with different resolutions in z.
"""
import sys
sys.path.insert(0, '/home/taylor/Documents/independent-research/modules/')
import dataDiscretization as dd
import scipy as sp
import numpy as np

z_discs = np.arange(10, 110, 10)
CHARGE = True
NOISE = True

for z_disc in z_discs:
    filename_p = "pDisc_40000_"+"20x20x" + str(z_disc) + ".npz"
    filename_C = "CDisc_40000_"+"20x20x" + str(z_disc) + ".npz"

    p_data = dd.bulkDiscretize('/home/taylor/Documents/independent-research/data/p_40000.h5', 20, 20, z_disc, CHARGE, NOISE)
    sp.sparse.save_npz('/home/taylor/Documents/independent-research/data/20x20xVaryingZ/' + filename_p, p_data)

    C_data = dd.bulkDiscretize('/home/taylor/Documents/independent-research/data/C_40000.h5', 20, 20, z_disc, CHARGE, NOISE)
    sp.sparse.save_npz('/home/taylor/Documents/independent-research/data/20x20xVaryingZ/' + filename_C, C_data)

print("Discretization complete.")
