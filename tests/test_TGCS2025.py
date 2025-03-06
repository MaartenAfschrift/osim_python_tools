import os
from muscle_redundancy_solver import muscle_redundancy_solver, ideal_muscles_actuated
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# # path to data (example data)
# mainpath = 'C:/Users/mat950/Documents/Software/general_tools/python_toolkit/osim_tools_python'
# osim_model_path = os.path.join(mainpath,'data','subject1.osim')
# ikfile = os.path.join(mainpath,'data','Walking_IK.mot')
# idfile = os.path.join(mainpath,'data','Walking_ID.sto')
tspan = [0.6, 1.9]

# path to data Koelewijn
datapath = 'E:\\Data\\Koelewijn2019\\RawData\\addbiomech\\Koelewijn_Subject11\\'
osim_model_path = os.path.join(datapath, 'Models', 'optimized_scale_and_markers.osim')
ikfolder = os.path.join(datapath, 'IK')
idfolder = os.path.join(datapath, 'IDm')
ikfile = os.path.join(ikfolder, 'Subject11_trial5_marker_ik.mot')
idfile = os.path.join(idfolder, 'Subject11_trial5_id.sto')
tspan = [20, 40]

dofs = ["ankle_angle_r"]
muscles = ['soleus_r','tib_ant_r']

# # muscle redundancy solver
my_mrs = muscle_redundancy_solver(osim_model_path, ikfile, idfile, dofs, muscles_selected=muscles)
#muscles_sel = my_mrs.identify_muscles()
my_mrs.filter_inputs(cutoff_frequency=6, tstart = tspan[0], tend = tspan[1])
my_mrs.formulate_solve_ocp(dt = 0.01, tstart = tspan[0], tend = tspan[1] )

# muscle redundancy solver with linear actuator as muscle fiber model
my_linact_solver = ideal_muscles_actuated(osim_model_path, ikfile, idfile, dofs, muscles_selected=muscles)
#muscles_sel = my_linact_solver.identify_muscles()
my_linact_solver.filter_inputs(cutoff_frequency=6, tstart = tspan[0], tend = tspan[1])
my_linact_solver.formulate_solve_ocp(dt = 0.01, tstart = tspan[0], tend = tspan[1] )

# plot muscle power
plt.figure()
for i in range(len(my_mrs.muscles_selected)):
    plt.subplot(2,1,i+1)
    plt.plot(my_mrs.solution['t'], my_mrs.solution['fiber_power'][i,:].T, label='mrs')
    plt.plot(my_linact_solver.solution['t'], my_linact_solver.solution['fiber_power'][i,:].T, label='linact')
plt.legend()

plt.figure()
for i in range(len(my_mrs.muscles_selected)):
    plt.subplot(2,1,i+1)
    plt.plot(my_mrs.solution['t'], my_mrs.solution['tendon_force'][i,:].T, label='mrs')
    plt.plot(my_linact_solver.solution['t'], my_linact_solver.solution['Ftendon'][i,:].T, label='linact')
plt.legend()

plt.figure()
for i in range(len(my_mrs.muscles_selected)):
    plt.subplot(2,1,i+1)
    plt.plot(my_mrs.solution['t'], my_mrs.solution['a'][i,:].T, label='mrs')
    plt.plot(my_linact_solver.solution['t'], my_linact_solver.solution['a'][i,:].T, label='linact')
plt.legend()

plt.figure()
for i in range(len(my_mrs.muscles_selected)):
    plt.subplot(2,1,i+1)
    plt.plot(my_mrs.solution['t'], my_mrs.solution['tendon_length'][i,:].T, label='mrs')
    plt.plot(my_linact_solver.solution['t'], my_linact_solver.solution['tendon_length'][i,:].T, label='linact')
plt.ylabel('tendon length [m]')
plt.legend()

plt.figure()
for i in range(len(my_mrs.muscles_selected)):
    plt.subplot(2,1,i+1)
    plt.plot(my_mrs.solution['t'], my_mrs.solution['fiber_power'][i,:].T, label='mrs')
    plt.plot(my_linact_solver.solution['t'], my_linact_solver.solution['fiber_power'][i,:].T, label='linact')
plt.ylabel('fiber power')
plt.legend()


plt.figure()
plt.plot(my_mrs.solution['t'], my_mrs.solution['tau_ideal'].T, label='mrs')
plt.plot(my_linact_solver.solution['t'], my_linact_solver.solution['tau_ideal'].T, label='linact')
plt.legend()

plt.figure()
plt.plot(my_mrs.solution['t'], my_mrs.solution['muscle_torque'].T, label='mrs')
plt.plot(my_linact_solver.solution['t'], my_linact_solver.solution['muscle_torque'].T, label='linact')
plt.legend()

plt.show()








