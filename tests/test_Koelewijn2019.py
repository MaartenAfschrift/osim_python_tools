# test some functionality of the osim_subject class

import opensim as osim
from osim_utilities import osim_subject
from kinematic_analyses import lmt_api, moment_arm_api
from muscle_redundancy_solver import muscle_redundancy_solver
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# I want to get the path to the main folder (osim_tools_python)
import os

# flow control
bool_compute_lmt = False
bool_compute_dm = False
bool_test_mrs = True

# path to datafiles
datapath = 'E:\\Data\\Koelewijn2019\\RawData\\addbiomech\\Koelewijn_Subject11\\'
modelpath = os.path.join(datapath, 'Models', 'optimized_scale_and_markers.osim')
ikfolder = os.path.join(datapath, 'IK')
idfolder = os.path.join(datapath, 'IDm')

# saving
lmt_folder = os.path.join(datapath, 'lmt')
dm_folder = os.path.join(datapath, 'dM')

# select specific muscles
muscles_sel = ['soleus_r','tib_ant_r','soleus_l','tib_ant_l']
dof_sel = ['ankle_angle_r','ankle_angle_l']

# init osim_subject model
subj = osim_subject(modelpath)
subj.set_ikfiles_fromfolder(ikfolder)

# compute muscle-tendon lengths and moment arms using api
# instead of muscle analysis (this is slow)
# set time window for analysis
tstart = 0 # start time (used for all motion files)
tend = 99 # end time (used for all motion files)
subj.set_lmt_folder(lmt_folder)
if bool_compute_lmt:
    subj.compute_lmt(tstart = tstart, tend= tend, selected_muscles= muscles_sel)
subj.set_momentarm_folder(dm_folder)
if bool_compute_dm:
    subj.compute_dM(tstart = tstart, tend = tend,
                    selected_muscles= muscles_sel,
                    selected_dofs= dof_sel)

# test muscle redundancy solver on a trial
if bool_test_mrs:
    # init mrs object
    dofs = ['ankle_angle_l']
    muscles = ['soleus_l','tib_ant_l']
    ikfile = os.path.join(ikfolder,'Subject11_trial5_marker_ik.mot')
    idfile = os.path.join(idfolder, 'Subject11_trial5_id.sto')
    tspan = [20, 24]
    # init mrs object
    my_mrs = muscle_redundancy_solver(modelpath, ikfile, idfile)
    my_mrs.set_dofs(dofs)
    my_mrs.set_muscles(muscles)
    # compute lmt and dm
    my_mrs.compute_lmt_dm(tstart= tspan[0], tend= tspan[1])
    my_mrs.filter_inputs(cutoff_frequency=6)
    # debug for lmt lengts
    my_mrs.debug_lmt()
    # solve static optimization
    so_results = my_mrs.run_static_optimization(dt = 0.005, tstart= tspan[0], tend= tspan[1])

    # solve muscle redundancy
    my_mrs.formulate_solve_ocp(dt=0.005, tstart=tspan[0], tend=tspan[-1], bool_static_opt= True)

    # plot activation static optimization
    plt.figure()
    plt.plot(so_results['t'], so_results['a'].T)
    plt.xlabel('time [s]')
    plt.ylabel('activation [-]')
    plt.legend(muscles)

    plt.figure()
    plt.plot(so_results['t'],so_results['topt'].T, label = 'torque actuator')
    plt.plot(my_mrs.t, my_mrs.id.T, label = 'inverse dynamics')
    plt.xlabel('time [s]')
    plt.ylabel('tau [Nm]')

    plt.figure()
    plt.plot(so_results['t'],so_results['lm_tilde'].T)
    plt.xlabel('time [s]')
    plt.ylabel('lm_tilde []')
    plt.legend(muscles)

    # plot optimal solution mrs
    plt.figure()
    plt.plot()
    ctm = -1
    for m in my_mrs.muscles_selected:
        ctm += 1
        plt.plot(my_mrs.solution['t'], my_mrs.solution['a'][ctm, :], label=m)
    plt.xlabel('time [s]')
    plt.ylabel('activation [-]')
    plt.legend()

    plt.figure()
    plt.plot(my_mrs.solution['t'], my_mrs.solution['muscle_torque'], label="muscles")
    plt.plot(my_mrs.solution['t'], my_mrs.solution['tau_ideal'], label='torque actuator')
    plt.plot(my_mrs.solution['t'], my_mrs.solution['id'][0, :], label='torque')
    plt.xlabel('time [s]')
    plt.ylabel('torque')
    plt.legend()

    plt.figure()
    plt.plot(my_mrs.solution['t'], my_mrs.solution['fiber_power'])
    plt.xlabel('time [s]')
    plt.ylabel('fiber power')

    plt.figure()
    plt.plot(my_mrs.solution['t'], my_mrs.solution['tendon_power'])
    plt.xlabel('time [s]')
    plt.ylabel('tendon power')





    plt.show()








