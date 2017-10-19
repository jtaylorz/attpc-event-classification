"""
Testing logistic regression on nuclear scattering data.
"""

import scipy as sp
import numpy as np
import sklearn

p_data = sp.sparse.load_npz('/home/taylor/Documents/independent-research/data/pDisc_40000.npz')
C_data = sp.sparse.load_npz('/home/taylor/Documents/independent-research/data/pDisc_40000.npz')
p_labels = np.zeros((p_data.shape[0],1))
C_labels = np.ones((C_data.shape[0],1))

print(p_data.shape)
print(p_labels.shape)
print(C_data.shape)
print(C_labels.shape)

full_data = sp.sparse.vstack([p_data, C_data], format='csr')
print(full_data.shape)
full_labels = np.vstack((p_labels, C_labels))
print(full_labels.shape)
