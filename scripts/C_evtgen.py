import math
import numpy as np

from effsim.paramgen import uniform_param_generator
from effsim.paramgen import distribution_param_generator
from effsim.effsim import EventSimulator
from effsim.effsim import NoiseMaker
from pytpc.hdfdata import HDFDataFile
import pytpc

import yaml
import h5py

with open('../req-files/config_e15503b_C.yml', 'r') as f:
    config = yaml.load(f)

beam_enu0 = config['beam_enu0']
beam_mass = config['beam_mass']
beam_charge = config['beam_charge']
mass_num = config['mass_num']
max_beam_angle = (config['max_beam_angle']*math.pi)/180
beam_origin_z = config['beam_origin_z']

gas = pytpc.gases.InterpolatedGas('isobutane', 19.2)

# number of events to create
num_evts = 40000

#adding 1000 event cusion for possibility of failed event sim
Cgen = uniform_param_generator(beam_enu0, beam_mass, beam_charge, mass_num, max_beam_angle, beam_origin_z, gas, num_evts+1000)

sim = EventSimulator(config)

with HDFDataFile('../data/tilt/C_40000_tilt.h5', 'w') as hdf:
    evt_id = 0
    for C in Cgen:
        if(evt_id > num_evts):
            break;
        else:
            try:
                evt, ctr = sim.make_event(C[0][0], C[0][1], C[0][2], C[0][3], C[0][4], C[0][5])
            except IndexError:
                print("Bad event, skipping")
                continue;

        pyevt = sim.convert_event(evt, evt_id)

        hdf.write_get_event(pyevt)
        print("Wrote event " + str(evt_id) + " with " + str(len(pyevt.traces)) + " traces")
        evt_id += 1

print(str(evt_id-1) + " events written to file")
