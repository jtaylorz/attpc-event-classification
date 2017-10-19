"""

"""
import math
import numpy as np
import scipy as sp

import pytpc

from pytpc.hdfdata import HDFDataFile
import yaml
import h5py


DETECTOR_LENGTH = 1000.0
DETECTOR_RADIUS = 275.0

def discretizeGrid(xyz, x_disc, y_disc, z_disc):
    """Discretizes AT-TPC point cloud data using a grid geometry.

    Parameters
    ----------
    xyz    : point cloud data with shape (n,5) where n is the number of traces
    x_disc : number of slices in x
    y disc : number of slices in y
    z_disc : number of slices in z

    Returns
    -------
    The discretized data in a csr sparse matrix of shape (1, x_disc*y_disc*z_disc)
    """

    #calculate desired discretization resolution
    discElements = x_disc*y_disc*z_disc

    #calculate dimensional increments
    x_inc = (2*DETECTOR_RADIUS)/x_disc
    y_inc = (2*DETECTOR_RADIUS)/y_disc
    z_inc = DETECTOR_LENGTH/z_disc

    buckets = []

    for point in xyz:
        x_bucket = math.floor(((point[0]+DETECTOR_RADIUS)/(2*DETECTOR_RADIUS))*x_disc)
        y_bucket = math.floor(((point[1]+DETECTOR_RADIUS)/(2*DETECTOR_RADIUS))*y_disc)
        z_bucket = math.floor((point[2]/DETECTOR_LENGTH)*z_disc)

        bucket_num = z_bucket*x_disc*y_disc + x_bucket + y_bucket*y_disc
        buckets.append(bucket_num)

    cols = np.unique(buckets)
    #rows = np.zeros(len(cols))
    #data = np.ones(len(cols))

    #discretized_data = sp.sparse.csr_matrix((data, (rows, cols)), shape=(1, discElements))
    #return discretized_data

    return cols


def bulkDiscretize(hdfPath, x_disc, y_disc, z_disc):
    """Discretizes all events in an HDF file using a grid geometry.

    Parameters
    ----------
    hdfPath : the system path to the hdf5 file to be
    x_disc  : number of slices in x
    y disc  : number of slices in y
    z_disc  : number of slices in z

    Returns
    -------
    A numpy array of shape (n, x_disc*y_disc*z_disc) where n is the number of
    events in the provided hdf5 file.
    """

    discElements = x_disc*y_disc*z_disc
    discEvts = []

    with pytpc.HDFDataFile(hdfPath, 'r') as f:
        n_evts = len(f)
        evt_id = 0


        while (evt_id < 1000):
            curEvt = f[evt_id]
            curxyz = curEvt.xyzs(peaks_only=True, return_pads=True, baseline_correction=False, cg_times=False)

            discEvts.append(discretizeGrid(curxyz, x_disc, y_disc, z_disc))

            if (evt_id%10 == 0):
                print("Discretized event " + str(evt_id))
            evt_id += 1

    discretized_data = []
    for evt in discEvts:
        rows = np.zeros(len(evt))
        data = np.ones(len(evt))

        discretized_data.append(sp.sparse.csr_matrix((data, (rows, evt)), shape=(1, discElements)))

    discretized_data_mat = sp.sparse.vstack(discretized_data, format='csr')

    print("Data discretization complete.")
    return discretized_data_mat
