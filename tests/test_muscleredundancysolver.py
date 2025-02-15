# simple script to test the muscle redundancy solver
#-----------------------------------------------------

import os
from muscle_redundancy_solver import muscle_redundancy_solver
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# path to data
mainpath = 'C:/Users/mat950/Documents/Software/general_tools/python_toolkit/osim_tools_python'
osim_model_path = os.path.join(mainpath,'data','subject1.osim')
ikfile = os.path.join(mainpath,'data','Walking_IK.mot')
idfile = os.path.join(mainpath,'data','Walking_ID.sto')


# create muscle redundancy solver object
my_mrs = muscle_redundancy_solver(osim_model_path, ikfile, idfile)

# set dofs
dofs = ['hip_flexion_r','knee_angle_r','ankle_angle_r']
my_mrs.set_dofs(dofs)

# identify muscles for selected dofs
muscles_sel = my_mrs.identify_muscles()
print(muscles_sel) # just print to test if this works

# test function to compute moment arms and muscle-tendon lengths for selected muscles and dofs
my_mrs.compute_lmt_dm()
my_mrs.filter_inputs()

# test function to filter all inputs
my_mrs.get_muscle_properties()









# plot some moment arms to check if everything is fine
plt.figure()
#plt.plot()
dm_dat = my_mrs.my_subject.dm_dat[0]
plt.plot(dm_dat.time,dm_dat.soleus_r_ankle_angle_r,label='soleus_r ankle angle')
plt.plot(dm_dat.time,dm_dat.soleus_r_knee_angle_r,label='soleus_r knee angle')
plt.plot(dm_dat.time,dm_dat.lat_gas_r_ankle_angle_r,label='gas_r ankle angle')
plt.plot(dm_dat.time,dm_dat.lat_gas_r_knee_angle_r,label='gas_r knee angle')
plt.plot(dm_dat.time,dm_dat.bifemlh_r_knee_angle_r,label='bifem knee angle')
plt.plot(dm_dat.time,dm_dat.bifemlh_r_hip_flexion_r,label='bifem hip flexion')
plt.plot(dm_dat.time,dm_dat.bifemlh_r_ankle_angle_r,label='bifem ankle angle')
plt.legend()


plt.show()







