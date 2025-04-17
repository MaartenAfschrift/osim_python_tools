

# add parent directory to path
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from muscle_redundancy_solver import (muscle_redundancy_solver,
                                      ideal_muscles_actuated,
                                      analyse_mrs_results)
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures
import pandas as pd
import numpy as np

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
dofs =['knee_angle_r' ,'ankle_angle_r','hip_flexion_r','hip_adduction_r','hip_rotation_r']
#dofs =['ankle_angle_r','knee_angle_r']
#dofs =['knee_angle_r','ankle_angle_r']
tspan = [0.6, 1.95]
my_mrs = muscle_redundancy_solver(osim_model_path, ikfile, idfile, dofs,
                                  outpath=os.path.join(mainpath,'data','mrs_minact_allmuscles'))
my_mrs.identify_muscles() # identifies all muscles that tpan the dofs
my_mrs.filter_inputs(cutoff_frequency=6, tstart = tspan[0], tend = tspan[1])
my_mrs.formulate_solve_ocp(dt = 0.01, tstart = tspan[0], tend = tspan[1],
                           objective_function='min_act')
my_mrs.default_plot()
my_mrs.plot_static_opt_results()
# function to analyse mrs results
analyse_mrs_results(my_mrs.solution)

# debugging problem with MRS
# as long I use one dof everything seems fine and things to wrong if I use multiple dofs
# potential problems:
#   - torque equilibrium with wrong id moment
#   - problem moment arm computation (w.r.t specific joint)

# there is clearly something wrong. now the muscles at the knee have a moment arm around the ankle joint

# sol = my_mrs.solution
# plt.figure()
# iknee = np.arange(0, 11)
# iankle = np.arange(11, 23)
# knee_muscles =  [my_mrs.muscles_selected[i] for i in iknee]
# ankle_muscles =  [my_mrs.muscles_selected[i] for i in iankle]
#
# plt.subplot(1,2,1)
# plt.plot(sol['t'], sol['moment_arm'][iknee,:,0].T)
# plt.legend(knee_muscles)
#
# plt.subplot(1,2,2)
# plt.plot(sol['t'], sol['moment_arm'][iankle,:,0].T)
# plt.legend(ankle_muscles)
# plt.title(my_mrs.dofs[0])
#
# plt.figure()
# plt.subplot(1,2,1)
# plt.plot(sol['t'], sol['moment_arm'][iknee,:,1].T)
# plt.legend(knee_muscles)
#
# plt.subplot(1,2,2)
# plt.plot(sol['t'], sol['moment_arm'][iankle,:,1].T)
# plt.legend(ankle_muscles)
# plt.title(my_mrs.dofs[1])


# rond knie lijkt alles goed te gaan, rond enkel gaat het lelijk mis als enkel eerst en dan knie als input
# identiek probleem als ik volgorde omgekeerd doe
# probleem zit niet in het berekenen van hefbomen (self.dm_dat is goed, bijvoorbeeld plt.plot(dm.soleus_r_ankle_angle_r))



# analyze simulation results (and compare it to matlab solution)



plt.show()
