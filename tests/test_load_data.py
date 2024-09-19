# import utilities
from osim_utilities import osim_subject
import os
import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
import pandas as pd
import opensim as osim
from inverse_dynamics import InverseDynamics
from inverse_kinematics import InverseKinematics
from kinematic_analyses import bodykinematics
from scipy import signal
import scipy.interpolate as interpolate

import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# test load data from pickle
mainpath = 'D:\Data\Berkelmans'
datapath = os.path.join(mainpath,'SUB04')
datafile = os.path.join(datapath, 'osim','fast_skatebelt_subj.pkl')

with open(datafile, 'rb') as f:
    subj = pickle.load(f)

print('test')
plt.figure()
ifile = 0
plt.plot(subj.ikdat[ifile].time, subj.ikdat[ifile].ankle_angle_r)
plt.title(subj.filenames[ifile])
