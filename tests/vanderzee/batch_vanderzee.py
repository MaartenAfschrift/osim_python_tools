from general_utilities import readMotionFile, WriteMotionFile, lowPassFilterDataFrame
from muscle_redundancy_solver import muscle_redundancy_solver, muscle_redundancy_solver_exo
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures
import os

# to do aanpassingen:
# only plantarflexion assistance
# only assistance during stance phase
# script for postprocessing (outcomes as computed by Lonit)



bool_default_mrs = True
bool_exo_shortening = True
bool_exo_percid = True

# Path information
gen_osim_model = ('C:/Users/mat950/Documents/Software/general_tools' +
                  '/python_toolkit/osim_tools_python/data/subject1.osim')
datapath = ('C:/Users/mat950/OneDrive - Vrije Universiteit Amsterdam' +
            '/Onderzoek/Data/VanDerZee')
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
speeds = [7, 9, 11, 14, 16, 18, 20]

# information MRS
dofs = ["ankle_angle_r"] # selected dofs
muscles = ['soleus_r','tib_ant_r'] # selected muscles
time_window = [10,20] # start and end time in seconds
achilles_tendon_stiffness = 30 # kT variable in DeGroote Muscle model
perc_assistance = 0.4 # assistance as ratio of the inverse dynamic ankle moment

for s in subjects:
    for v in speeds:
        prefix = 'subject_' + str(s) + '_' + str(v)
        ikfile_mot = os.path.join(datapath,'dynamics_processingpass', (prefix + '_IK_all.mot'))
        idfile_mot = os.path.join(datapath,'dynamics_processingpass', (prefix + '_ID_all.sto'))

        if bool_default_mrs:
            print('Running default MRS for subject', s, 'at speed', v)
            out_path = os.path.join(datapath,'mrs_results','default')
            my_mrs = muscle_redundancy_solver(gen_osim_model, ikfile_mot,
                                              idfile_mot, dofs, muscles, outpath=out_path)
            my_mrs.filter_inputs(cutoff_frequency=6)
            my_mrs.set_tendon_stiffness('soleus_r', achilles_tendon_stiffness)
            my_mrs.formulate_solve_ocp(dt=0.01, tstart=time_window[0], tend=time_window[1])
            # delete my_mrs object to save memory
            del my_mrs
        if bool_exo_percid:
            print('Running exoskeleton MRS with percentage ID control for subject', s, 'at speed', v)
            out_path = os.path.join(datapath,'mrs_results','perc_id')
            my_mrs = muscle_redundancy_solver_exo(gen_osim_model, ikfile_mot,
                                              idfile_mot, dofs, muscles, outpath=out_path)
            my_mrs.filter_inputs(cutoff_frequency=6)
            my_mrs.set_tendon_stiffness('soleus_r', achilles_tendon_stiffness)
            my_mrs.set_controller_type('percentage_id')
            my_mrs.set_percentage_id_assistance(perc_assistance)

            my_mrs.formulate_solve_ocp(dt=0.01, tstart=time_window[0], tend=time_window[1])

            del my_mrs
        if bool_exo_shortening:
            print('Running exoskeleton MRS with percentage ID shortening control for subject', s, 'at speed', v)
            out_path = os.path.join(datapath, 'mrs_results', 'perc_id_shortening')
            my_mrs = muscle_redundancy_solver_exo(gen_osim_model, ikfile_mot,
                                                  idfile_mot, dofs, muscles, outpath=out_path)
            my_mrs.filter_inputs(cutoff_frequency=6)
            my_mrs.set_tendon_stiffness('soleus_r', achilles_tendon_stiffness)
            my_mrs.set_controller_type('percentage_id_shortening')
            my_mrs.set_percentage_id_assistance(perc_assistance)

            my_mrs.formulate_solve_ocp(dt=0.01, tstart=time_window[0], tend=time_window[1])
            del my_mrs








