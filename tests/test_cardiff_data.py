# test batch processing on cardiff data
#---------------------------------------

# import utilities
from osim_utilities import osim_subject, readMotionFile
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

matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# path information (this should be everything :))
mainpath = 'C:/Users/mat950/Documents/Data/Cardiff_Lonit/osimData_young'
datapath = os.path.join(mainpath,'Patient2')
model_path = os.path.join(datapath, 'model','scaled_model_marker.osim')
trc_folder = os.path.join(datapath, 'RefWalk', 'data')
grf_folder = os.path.join(datapath, 'RefWalk', 'data')
ik_folder = os.path.join(datapath, ' RefWalk', 'ik_vx')
id_folder = os.path.join(datapath, 'RefWalk', 'ID_vx')
lmt_folder = os.path.join(datapath, 'RefWalk', 'lmt')
dm_folder = os.path.join(datapath, 'RefWalk', 'dM')
general_ik_settings = os.path.join(mainpath, 'osim_settings','IK_settings.xml')
general_id_settings = os.path.join(mainpath, 'osim_settings','ID_settings.xml')
loads_settings = os.path.join(mainpath, 'osim_settings','loads_settings.xml')

# create object for processing
subj = osim_subject(model_path)

# set all the trc files
subj.set_trcfiles_fromfolder(trc_folder)

# run inverse kinematics
subj.set_ik_directory(ik_folder)
subj.set_general_ik_settings(general_ik_settings)
subj.compute_inverse_kinematics(overwrite= False)

# run inverse dynamics
subj.set_ext_loads_dir(grf_folder)
subj.set_id_directory(id_folder)
subj.compute_inverse_dynamics()

# compute muscle-tendon lengths and moment arms using api
tstart = 0.5 # start time (used for all motion files)
tend = 5 # end time (used for all motion files)
subj.set_lmt_folder(lmt_folder)
subj.compute_lmt(tstart = tstart, tend= tend)
subj.set_momentarm_folder(dm_folder)
subj.compute_dM(tstart = tstart, tend = tend)

# notes for Lonit:
# you can use utilsTRC to adapt the trc file
# you can use readMotionFile and generate_mot_file to read and write the grf file



