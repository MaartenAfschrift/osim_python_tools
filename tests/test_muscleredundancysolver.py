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

# currently debugging case with knee angle. there is clearly something wrong here
#   - maybe passive forces ?

# create muscle redundancy solver object
my_mrs = muscle_redundancy_solver(osim_model_path, ikfile, idfile)

# set dofs
dofs =['hip_flexion_r','knee_angle_r' ,'ankle_angle_r','hip_adduction_r','hip_rotation_r']
#dofs =['knee_angle_r' ,'ankle_angle_r']
#dofs = ["ankle_angle_r"]
#dofs = ["knee_angle_r"]
my_mrs.set_dofs(dofs)

# identify muscles for selected dofs
muscles_sel = my_mrs.identify_muscles()

# set muscles
#muscles = ['soleus_r','tib_ant_r']
#muscles = ['bifemsh_r','vas_med_r', 'vas_int_r', 'vas_lat_r','med_gas_r']
#muscles = ['vas_med_r', 'vas_int_r']
#my_mrs.set_muscles(muscles)

# test function to compute moment arms and muscle-tendon lengths for selected muscles and dofs
my_mrs.compute_lmt_dm()
my_mrs.filter_inputs(cutoff_frequency=6)

# test function to get muscle properties
my_mrs.get_muscle_properties()

# debug lmt lengths
my_mrs.debug_lmt()

# test formulate and solve ocp
my_mrs.formulate_solve_ocp(dt = 0.01, t0 = 0.6,tend = 1.9 )

# plot optimal solution
plt.figure()
plt.plot()
ctm = -1
for m in my_mrs.muscles_selected:
    ctm += 1
    plt.plot(my_mrs.solution['t'], my_mrs.solution['a'][ctm,:],label=m)
plt.xlabel('time [s]')
plt.ylabel('activation [-]')
plt.legend()

# plot optimal solution
plt.figure()
plt.plot()
ctm = -1
for m in my_mrs.muscles_selected:
    ctm += 1
    plt.plot(my_mrs.solution['t'], my_mrs.solution['lm_tilde'][ctm,:],label=m)
plt.xlabel('time [s]')
plt.ylabel('lm tilde [-]')
plt.legend()


# plot optimal solution
plt.figure()
plt.plot()
ctm = -1
for m in my_mrs.muscles_selected:
    ctm += 1
    plt.plot(my_mrs.solution['t'], my_mrs.solution['vm_tilde'][ctm,:],label=m)
plt.xlabel('time [s]')
plt.ylabel('vm tilde [-]')
plt.legend()


plt.figure()
plt.plot()
ctdof = -1
ndofs = len(my_mrs.dofs)
if ndofs == 1:
    plt.plot(my_mrs.solution['t'], my_mrs.solution['muscle_torque'], label="muscles")
    plt.plot(my_mrs.solution['t'], my_mrs.solution['tau_ideal'], label='torque actuator')
    plt.plot(my_mrs.solution['t'], my_mrs.solution['id'][0,:], label='torque')
    plt.xlabel('time [s]')
    plt.ylabel('torque')
    plt.legend()
else:
    for d in my_mrs.dofs:
        ctdof = ctdof + 1
        plt.subplot(1, ndofs, ctdof+1)
        plt.plot(my_mrs.solution['t'], my_mrs.solution['muscle_torque'][ctdof,:],label="muscles")
        plt.plot(my_mrs.solution['t'],my_mrs.solution['tau_ideal'][ctdof,:],label='torque actuator')
        plt.plot(my_mrs.solution['t'],my_mrs.solution['id'][ctdof,:],label='torque')
        plt.xlabel('time [s]')
        plt.ylabel('torque')
    plt.legend()


plt.show()













# # plot some moment arms to check if everything is fine
# plt.figure()
# #plt.plot()
# dm_dat = my_mrs.my_subject.dm_dat[0]
# plt.plot(dm_dat.time,dm_dat.soleus_r_ankle_angle_r,label='soleus_r ankle angle')
# plt.plot(dm_dat.time,dm_dat.soleus_r_knee_angle_r,label='soleus_r knee angle')
# plt.plot(dm_dat.time,dm_dat.lat_gas_r_ankle_angle_r,label='gas_r ankle angle')
# plt.plot(dm_dat.time,dm_dat.lat_gas_r_knee_angle_r,label='gas_r knee angle')
# plt.plot(dm_dat.time,dm_dat.bifemlh_r_knee_angle_r,label='bifem knee angle')
# plt.plot(dm_dat.time,dm_dat.bifemlh_r_hip_flexion_r,label='bifem hip flexion')
# plt.plot(dm_dat.time,dm_dat.bifemlh_r_ankle_angle_r,label='bifem ankle angle')
# plt.legend()
#
#
# plt.show()







