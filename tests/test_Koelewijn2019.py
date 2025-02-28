# test some functionality of the osim_subject class

import opensim as osim
from osim_utilities import osim_subject
from kinematic_analyses import lmt_api, moment_arm_api
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# I want to get the path to the main folder (osim_tools_python)
import os

# path to datafiles
datapath = 'E:\\Data\\Koelewijn2019\\RawData\\addbiomech\\Koelewijn_Subject11\\'
modelpath = os.path.join(datapath, 'Models', 'optimized_scale_and_markers.osim')
ikfolder = os.path.join(datapath, 'IK')

# saving
lmt_folder = os.path.join(datapath, 'lmt')
dm_folder = os.path.join(datapath, 'dM')

# select specific muscles
muscles_sel = ['soleus_r','tib_ant_r','soleus_l','tib_ant_l']

# init osim_subject model
subj = osim_subject(modelpath)
subj.set_ikfiles_fromfolder(ikfolder)

# compute muscle-tendon lengths and moment arms using api
# instead of muscle analysis (this is slow)
# set time window for analysis
tstart = 0 # start time (used for all motion files)
tend = 99 # end time (used for all motion files)
subj.set_lmt_folder(lmt_folder)
#subj.compute_lmt(tstart = tstart, tend= tend, selected_muscles= muscles_sel)
subj.set_momentarm_folder(dm_folder)
subj.compute_dM(tstart = tstart, tend = tend, selected_muscles= muscles_sel)



