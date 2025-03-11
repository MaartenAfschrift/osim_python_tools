

# add parent directory to path
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
# datapath = 'E:\\Data\\Koelewijn2019\\RawData\\addbiomech\\Koelewijn_Subject11\\'
# osim_model_path = os.path.join(datapath, 'Models', 'optimized_scale_and_markers.osim')
# ikfolder = os.path.join(datapath, 'IK')
# idfolder = os.path.join(datapath, 'IDm')
# ikfile = os.path.join(ikfolder, 'Subject11_trial5_marker_ik.mot')
# idfile = os.path.join(idfolder, 'Subject11_trial5_id.sto')

# test on example data
mainpath = 'C:/Users/mat950/Documents/Software/general_tools/python_toolkit/osim_tools_python'
osim_model_path = os.path.join(mainpath,'data','subject1.osim')
ikfile = os.path.join(mainpath,'data','Walking_IK.mot')
idfile = os.path.join(mainpath,'data','Walking_ID.sto')

# # muscle redundancy solver with linear actuator as muscle fiber model
dofs =['hip_flexion_r','knee_angle_r' ,'ankle_angle_r','hip_adduction_r','hip_rotation_r']
tspan = [0.6, 1.95]
my_linact_solver = ideal_muscles_actuated(osim_model_path, ikfile, idfile, dofs,
                                          outpath=os.path.join(mainpath, 'data' ,'mrs_linact_allmuscles'))
muscles_sel = my_linact_solver.identify_muscles()
my_linact_solver.filter_inputs(cutoff_frequency=6, tstart = tspan[0], tend = tspan[1])
my_linact_solver.formulate_solve_ocp(dt = 0.01, tstart = tspan[0], tend = tspan[1],
                                     objective_function='min_act')
my_linact_solver.default_plot()
my_linact_solver.plot_static_opt_results()

plt.show()








