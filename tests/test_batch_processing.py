# test batch processing
#-----------------------

# import utilities
from osim_utilities import osim_subject
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# test with simone her dataset
datapath = 'C:/Users/mat950/Documents/Data/Berkelmans/Test_2406'
model_path = os.path.join(datapath, 'scaled_model_pp5.osim')
trc_folder = os.path.join(datapath, 'trc','fast_skatebelt')
ik_folder = os.path.join(datapath, 'ik', 'fast_skatebelt')

# some general settings
general_ik_settings = os.path.join(datapath, 'osim_settings', 'ik_settings.xml')

# create object for processing
subj = osim_subject(model_path)

# run inverse kinematics
subj.set_trcfiles_fromfolder(trc_folder)
subj.set_general_ik_settings(general_ik_settings)
subj.set_ik_directory(ik_folder)
subj.compute_inverse_kinematics()

#
print('test finished')


