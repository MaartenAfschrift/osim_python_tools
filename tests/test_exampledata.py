# test some functionality of the osim_subject class

import opensim as osim
from osim_utilities import osim_subject
from kinematic_analyses import lmt_api, moment_arm_api
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# I want to get the path to the main folder (osim_tools_python)
from pathlib import Path

# get current directory
main_dir = Path.cwd().parent
outputdir = Path.joinpath(main_dir,'data', 'output')

# path to datafiles
modelpath = str(Path.joinpath(main_dir, 'data', 'subject1.osim'))
idfile = str(Path.joinpath(main_dir, 'data', 'Walking_ID.sto'))
ikfile = str(Path.joinpath(main_dir, 'data', 'Walking_IK.mot'))

# select specific muscles
muscles_sel = ['soleus_r','tib_ant_r','soleus_l','tib_ant_l']

# lmt_object
lmt_obj = lmt_api(modelpath, ikfile, outputdir)
lmt_obj.compute_lmt(tstart = 0, tend =99, selected_muscles = muscles_sel)

# moment arm object
dm_obj = moment_arm_api(modelpath, ikfile, outputdir)
dm_dat = dm_obj.compute_dm(selected_muscles= muscles_sel)

# plot moment arm
plt.figure()
plt.plot(dm_dat[0].time, dm_dat[0].soleus_r_ankle_angle_r)
plt.plot(dm_dat[0].time, dm_dat[0].tib_ant_r_ankle_angle_r)
plt.show()






