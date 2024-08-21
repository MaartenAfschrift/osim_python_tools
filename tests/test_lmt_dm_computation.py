# import utilities
from osim_utilities import osim_subject
import os
import matplotlib.pyplot as plt

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

# test identify relevant dofs
my_subject.identify_relevant_dofs_dM()

# test compute muscle tendon lengths
#my_subject.compute_lmt()
my_subject.compute_dM()

print('test')