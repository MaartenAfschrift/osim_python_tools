
# import utilities
from osim_utilities import osim_subject
from general_utilities import readMotionFile, WriteMotionFile
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
from utilsTRC import TRCFile, trc_2_dict
from scipy import signal

matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures
# plt.ion()

# path information (this should be everything :))
mainpath = 'D:\Data\DataIsraelLuisPena\sub1'
datapath = os.path.join(mainpath,'S1_GC1')

# loading
model_path = os.path.join(mainpath,'1_scaledH_hammer2010.osim')
ik_folder = datapath
id_folder = datapath

# saving
lmt_folder = os.path.join(datapath, 'lmt')
dm_folder = os.path.join(datapath, 'dM')


#-----------------------------------------------
# ---               Processing
#-----------------------------------------------

# create object for processing
subj = osim_subject(model_path)
#subj.set_ik_directory(ik_folder)
subj.set_ikfiles_fromfolder(ik_folder)
#subj.set_id_directory(id_folder)

# compute muscle-tendon lengths and moment arms using api
# instead of muscle analysis (this is slow)
# set time window for analysis
tstart = 0.5 # start time (used for all motion files)
tend = 1 # end time (used for all motion files)
subj.set_lmt_folder(lmt_folder)
subj.compute_lmt(tstart = tstart, tend= tend)
subj.set_momentarm_folder(dm_folder)
subj.compute_dM(tstart = tstart, tend = tend)