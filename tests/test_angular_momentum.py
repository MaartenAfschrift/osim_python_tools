
# test computation angular momentum
#-----------------------------------

# import utilities
from osim_utilities import osim_subject
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures


# path to data
datapath_main = 'C:/Users/mat950/Documents/Data/DebugOsimCode'

# init an opensim subject
osim_model = os.path.join(datapath_main, 'subject01.osim')
datapath = datapath_main
my_subject = osim_subject(osim_model, maindir= datapath)

# test set a specific ik file
ikfile = os.path.join(datapath,'subject01_walk1_ik.mot')
my_subject.set_ikfiles(ikfile)
my_subject.read_ikfiles()

# compute angular momentum
my_subject.set_angmom_folder(os.path.join(datapath_main,'angmom'))
my_subject.compute_angular_momentum()

# plot angular momentum
plt.figure()
plt.plot(my_subject.angmom[0].time,my_subject.angmom[0].Lx)
plt.show()


