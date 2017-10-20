"""
dataDiscretization.py
=====================

This module contains functions for discretizing 3D point cloud data produced by
the Active-Target Time Projection Chamber.

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

def discretizeCylindrical(xyz, z_disc, radial_disc, angular_disc):
    """(Inefficiently) Discretizes AT-TPC point cloud data using a cylindrical
    geometry. We found that this strategy forces an inappropriate geometry on
    our data.

    Parameters
    ----------
    xyz          : point cloud data with shape (n,5) where n is the number of traces
    z_disc       : number of slices in z
    radial_disc  : number of radial slices/concentric circles
    angular_disc : number of angular wedge slices

    Returns
    -------
    The discretized data in an array of shape (1, z_disc*radial_disc*angular_disc)
    """
    #calculate dimensional increments
    z_inc = DETECTOR_LENGTH/z_disc
    radial_inc = DETECTOR_RADIUS/radial_disc
    angular_inc = (2*math.pi)/angular_disc

    #create slice boundary arrays
    z_slices = np.arange(DETECTOR_LENGTH, 0.0-z_inc, -z_inc)
    radial_slices = np.arange(DETECTOR_RADIUS, 0.0-radial_inc, -radial_inc)
    angular_slices = np.arange(-math.pi, math.pi+angular_inc, angular_inc)


    discretized_data = np.zeros((1,z_disc*radial_disc*angular_disc))
    discretized_xyz = np.zeros([xyz.shape[0],xyz.shape[1]])
    bucket_num = 0
    num_pts = 0

    for i in range(len(z_slices)-1):
        for j in range(len(radial_slices)-1):
            for k in range(len(angular_slices)-1):
                for point in xyz:
                    if ((z_slices[i] > point[2] > z_slices[i+1]) and
                    (radial_slices[j] > math.sqrt(point[0]**2+ point[1]**2) > radial_slices[j+1]) and
                    (angular_slices[k] < math.atan2(point[1], point[0])  < angular_slices[k+1])):

                        discretized_data[0][bucket_num] = 1
                        num_pts += 1

            bucket_num += 1

    return discretized_data


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

        bucket_num = z_bucket*x_disc*y_disc + x_bucket + y_bucket*x_disc
        buckets.append(bucket_num)

    cols = np.unique(buckets)
    rows = np.zeros(len(cols))
    data = np.ones(len(cols))

    discretized_data = sp.sparse.csr_matrix((data, (rows, cols)), shape=(1, discElements))
    return discretized_data


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

        while (evt_id < n_evts):
            curEvt = f[evt_id]
            curxyz = curEvt.xyzs(peaks_only=True, return_pads=False, baseline_correction=False, cg_times=False)
            #pass first 3 coordinates
            discEvts.append(discretizeGrid(curxyz, x_disc, y_disc, z_disc))
            if (evt_id%1000 == 0):
                print("Discretized event " + str(evt_id))
            evt_id += 1

    discretized_data = sp.sparse.vstack(discEvts, format='csr')
    print("Data discretization complete.")
    return discretized_data
