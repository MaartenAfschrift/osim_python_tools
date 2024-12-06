# test batch processing on cardiff data
#---------------------------------------
from PyQt5.uic.properties import bool_

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

# path information (this should be everything :))
mainpath = 'C:/Users/mat950/Documents/Data/Cardiff_Lonit/osimData_young'
datapath = os.path.join(mainpath,'Patient2')
model_path = os.path.join(datapath, 'model','scaled_model_marker.osim')
trc_folder = os.path.join(datapath, 'RefWalk', 'data')
trc_folder_filter = os.path.join(datapath, 'RefWalk', 'data_filt')
grf_folder = os.path.join(datapath, 'RefWalk', 'data')
ik_folder = os.path.join(datapath, 'RefWalk', 'ik_vx')
id_folder = os.path.join(datapath, 'RefWalk', 'ID_vx2')
lmt_folder = os.path.join(datapath, 'RefWalk', 'lmt_test')
dm_folder = os.path.join(datapath, 'RefWalk', 'dM_test')
general_ik_settings = os.path.join(mainpath, 'osim_settings','IK_settings.xml')
general_id_settings = os.path.join(mainpath, 'osim_settings','ID_settings.xml')
loads_settings = os.path.join(mainpath, 'osim_settings','loads_settings.xml')


bool_filter = False

if bool_filter:
    #-----------------------------------------------
    # ---          Filter trc file
    #----------------------------------------------
    # adapt trc file
    trc = trc_2_dict(os.path.join(trc_folder,'Refwalk_marker.trc'), rotation=None)

    # there is a problem in the original trc files when t>=100s. It seems that it writes 100.005s as 100.01
    time = trc['time']
    isel = np.where(np.diff(time) == 0)
    time[isel] = time[isel]-0.005

    # low pass filter markers
    camera_rate = np.round(1./np.nanmean(np.diff(time)))
    order = 2
    cutoff = 6 / (camera_rate*0.5)
    b, a = signal.butter(order, cutoff, btype='low')
    for msel in trc['marker_names']:
        dat = trc['markers'][msel]
        # older problem with if marker is missing = 0, now should be nan
        imissing = np.where(dat[:, 0] == 0)[0]
        dat[imissing,:] = np.nan
        trc['markers'][msel] = signal.filtfilt(b, a, dat.T).T

    # write trc file
    nfr = len(time)
    trcfile = TRCFile(data_rate=camera_rate, camera_rate=camera_rate, num_frames=nfr, num_markers=0, units='m',
                  orig_data_rate=camera_rate, orig_data_start_frame=1, orig_num_frames=nfr, time=time)
    # add markers
    for msel in trc['marker_names']:
        # add marker to trc object
        marker = trc['markers'][msel]
        trcfile.add_marker(msel, marker[:,0], marker[:,1], marker[:,2])

    # write trc file
    if not os.path.exists(trc_folder_filter):
        os.makedirs(trc_folder_filter)
    trcfile.write(os.path.join(trc_folder_filter, 'Refwalk_marker.trc'))

    #-----------------------------------------------
    # ---          Filter grf file
    #----------------------------------------------
    grf = readMotionFile(os.path.join(trc_folder,'Refwalk_GRF.mot'))
    sf = np.round(1/np.mean(np.diff(grf.time)))
    labels = grf.columns
    cutoff = 6/(sf*0.5)
    order = 2
    b, a = signal.butter(order, cutoff, btype='low')
    for lab in labels:
        if lab!= 'time':
            grf[lab] = signal.filtfilt(b,a, grf[lab])

    WriteMotionFile(grf.to_numpy(), labels, os.path.join(trc_folder_filter,'Refwalk_GRF.mot'))





#-----------------------------------------------
# ---           Batch processing
#----------------------------------------------


# create object for processing
subj = osim_subject(model_path)

# set all the trc files
subj.set_trcfiles_fromfolder(trc_folder_filter)

# run inverse kinematics
subj.set_ik_directory(ik_folder)
subj.set_general_ik_settings(general_ik_settings)
subj.compute_inverse_kinematics(overwrite= False)

# run inverse dynamics
subj.set_generic_external_loads(loads_settings)
subj.set_folder_grfdat(trc_folder_filter)
subj.set_id_directory(id_folder)
subj.set_general_id_settings(general_id_settings)
subj.compute_inverse_dynamics()

# compute muscle-tendon lengths and moment arms using api
# instead of muscle analysis (this is slow)
# set time window for analysis
tstart = 0.5 # start time (used for all motion files)
tend = 5 # end time (used for all motion files)
subj.set_lmt_folder(lmt_folder)
subj.compute_lmt(tstart = tstart, tend= tend)
subj.set_momentarm_folder(dm_folder)
subj.compute_dM(tstart = tstart, tend = tend)


