import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import numpy as np

from muscle_redundancy_solver import muscle_redundancy_solver, ideal_muscles_actuated, analyse_mrs_results
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures
import pandas as pd

#plt.ion()

# ---------------------------------------
#               path information
# ---------------------------------------

main_datapath = 'C:\\Users\\mat950\\Documents\\Data\\Cardiff_Lonit\\osimData_young'
out_path_main = 'C:\\Users\\\mat950\\\Documents\\\Data\\\Sim_TGCS\\\CardiffData'

subject =  'Patient10'
osim_model_path = os.path.join(main_datapath, subject, 'model','scaled_model_marker.osim')
ikfile = os.path.join(main_datapath, subject, 'RefWalk', 'KS','KS_Refwalk.mot')
idfile = os.path.join(main_datapath, subject, 'RefWalk', 'ID','ID_Refwalk.sto')
out_path = os.path.join(out_path_main, subject + '_test_energy3')
base_filename = 'Refwalk'

# ---------------------------------------
#           muscle redundancy solver
# ---------------------------------------
tspan = [30, 34]
timestep = 0.005
dofs = ['knee_angle_r', 'ankle_angle_r', 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r']
dofs = ['ankle_angle_r']
#muscles_sel = ['soleus_r','tib_ant_r',]

#-----------------------------------------------------------------------------------------
# Create objects for muscle redundancy solver and for linear actuator as muscle fiber model
#-----------------------------------------------------------------------------------------
my_mrs = muscle_redundancy_solver(osim_model_path, ikfile, idfile, dofs,
                                  outpath=out_path)
my_mrs.identify_muscles()
#my_mrs.set_muscles(muscles_sel)
my_mrs.filter_inputs(cutoff_frequency=4, tstart=tspan[0], tend=tspan[1])

my_mrs.set_filename(base_filename + '_mrs_minact')
my_mrs.formulate_solve_ocp(dt=timestep, tstart=tspan[0], tend=tspan[1],
                           objective_function='min_act',
                           opt_var_info='default')
my_mrs.default_plot()
my_mrs.plot_static_opt_results()
analysis_mrs = analyse_mrs_results(my_mrs.solution)



# simpler example with linear actuator
# muscle redundancy solver with linear actuator as muscle fiber model
my_linact_solver = ideal_muscles_actuated(osim_model_path, ikfile, idfile, dofs,
                                          outpath=out_path)
muscles_sel = my_linact_solver.identify_muscles()
#my_linact_solver.set_muscles(muscles_sel)
my_linact_solver.filter_inputs(cutoff_frequency=4, tstart=tspan[0], tend=tspan[1])
my_linact_solver.set_filename(base_filename + '_linact_minact')
my_linact_solver.formulate_solve_ocp(dt=timestep, tstart=tspan[0], tend=tspan[1],
                                     objective_function='min_act')

my_linact_solver.default_plot()
analysis_linact = analyse_mrs_results(my_linact_solver.solution)
#
# my_linact_solver.formulate_solve_ocp(dt=timestep, tstart=tspan[0], tend=tspan[1],
#                                      objective_function='min_fiber_power_squared',
#                                      opt_var_info= 'default')
# my_linact_solver.default_plot()
# analysis_linact_minP = analyse_mrs_results(my_linact_solver.solution)


# print some solutions
print(np.sum(analysis_mrs['fiberwork']))
print(np.sum(analysis_linact['fiberwork']))
# print(np.sum(analysis_linact_minP['fiberwork']))


plt.show()