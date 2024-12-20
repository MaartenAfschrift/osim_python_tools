# import utilities
from osim_utilities import osim_subject
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures


# path to data
datapath_main = 'C:/Users/mat950/Documents/Data/Cardiff_Lonit/osimData_young/Patient2'

# init an opensim subject
osim_model = os.path.join(datapath_main, 'model', 'scaled_model_marker_osim45.osim')
datapath = os.path.join(datapath_main, 'RefWalk')
my_subject = osim_subject(osim_model, maindir= datapath)

# set ik files
#ik_folder = os.path.join(datapath, 'KS')
#my_subject.set_ikfiles_fromfolder(ik_folder)

# test set a specific ik file
ikfile = os.path.join(datapath, 'KS', 'KS_Refwalk.mot')
my_subject.set_ikfiles(ikfile)
my_subject.read_ikfiles()

# test compute muscle tendon lengths
my_subject.set_lmt_folder(os.path.join(datapath,'lmt_debug'))
my_subject.set_momentarm_folder(os.path.join(datapath,'momentarm_debug'))
my_subject.compute_lmt(tstart = 10, tend = 12)
my_subject.compute_dM(tstart = 10, tend = 12)

lmt = my_subject.lmt_dat[0]
dm = my_subject.dm_dat[0]

plt.figure()
plt.subplot(1,2,1)
plt.plot(lmt.med_gas_r)
plt.subplot(1,2,2)
plt.plot(dm.med_gas_r_knee_angle_r)
plt.plot(dm.med_gas_r_ankle_angle_r)



print('test')