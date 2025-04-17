import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from muscle_redundancy_solver import muscle_redundancy_solver, ideal_muscles_actuated
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures
import pandas as pd

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
tspan = [20, 26]

dofs =['knee_angle_r' ,'ankle_angle_r','hip_flexion_r','hip_adduction_r','hip_rotation_r']

# default muscle redundancy solver
my_mrs = muscle_redundancy_solver(osim_model_path, ikfile, idfile, dofs,
                                  outpath=os.path.join(datapath,'mrs_TGCS'))
my_mrs.identify_muscles()
my_mrs.filter_inputs(cutoff_frequency=6, tstart = tspan[0], tend = tspan[1])
my_mrs.set_filename('mrs_minact_trial5')
my_mrs.formulate_solve_ocp(dt = 0.01, tstart = tspan[0], tend = tspan[1],
                           objective_function='min_act')

my_mrs.set_filename('mrs_minP_trial5')
my_mrs.formulate_solve_ocp(dt = 0.01, tstart = tspan[0], tend = tspan[1],
                           objective_function='min_pos_fiber_power')


# muscle redundancy solver with linear actuator as muscle fiber model
my_linact_solver = ideal_muscles_actuated(osim_model_path, ikfile, idfile, dofs,
                                          outpath=os.path.join(datapath,'mrs_linact'))
muscles_sel = my_linact_solver.identify_muscles()
my_linact_solver.filter_inputs(cutoff_frequency=6, tstart = tspan[0], tend = tspan[1])
my_linact_solver.set_filename('linact_minact_trial5')
my_linact_solver.formulate_solve_ocp(dt = 0.01, tstart = tspan[0], tend = tspan[1],
                           objective_function='min_act')

my_linact_solver.set_filename('linact_minP_trial5')
my_linact_solver.formulate_solve_ocp(dt = 0.01, tstart = tspan[0], tend = tspan[1],
                           objective_function='min_pos_fiber_power')

plt.show()








