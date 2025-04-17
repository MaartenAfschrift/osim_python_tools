# mrs analysis on data from Wang et al. 2023 [Twente dataset]
#-------------------------------------------------------------
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from muscle_redundancy_solver import muscle_redundancy_solver, ideal_muscles_actuated, \
    ideal_muscles_actuated_opt_tendon_stiffness
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures
import pandas as pd


main_datapath = 'E:\\Data\\Wang2023\\Processed_data\\Processed_data'
osim_model_path = os.path.join(main_datapath,'Subj04','OS','UTmodel',
                               'gait2392_simbody_subj04_scaled.osim')
os_datafolder = os.path.join(main_datapath,'Subj04','OS','DataFiles')

speeds = [18, 27, 36, 45, 54]
base_filename = 'Subj04walk'

for i in range(len(speeds)):
    # path to ik file
    ikfile = os.path.join(os_datafolder, base_filename + '_' + str(speeds[i]) + 'IK.mot')
    idfile = os.path.join(os_datafolder, base_filename + '_' + str(speeds[i]) + 'ID.sto')
    out_path = os.path.join(main_datapath,'Subj04','OS','mrs_TGCS')
    tspan  = [20, 26]
    dofs = ['knee_angle_r', 'ankle_angle_r', 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r']

    # muscle redundancy solver with linear actuator as muscle fiber model
    my_linact_solver = ideal_muscles_actuated_opt_tendon_stiffness(osim_model_path,
                                                                   ikfile, idfile, dofs,
                                                                   outpath=out_path)
    muscles_sel = my_linact_solver.identify_muscles()
    my_linact_solver.filter_inputs(cutoff_frequency=6, tstart=tspan[0], tend=tspan[1])
    my_linact_solver.set_filename(base_filename + '_' + str(speeds[i]) + '_linact_optkT_minact')
    my_linact_solver.formulate_solve_ocp(dt=0.01, tstart=tspan[0], tend=tspan[1],
                                         objective_function='min_act')

    my_linact_solver.set_filename(base_filename + '_' + str(speeds[i]) + '_linact_optkT_minP')
    my_linact_solver.formulate_solve_ocp(dt=0.01, tstart=tspan[0], tend=tspan[1],
                                         objective_function='min_pos_fiber_power')
