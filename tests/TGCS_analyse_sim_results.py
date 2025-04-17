# analyse simulation results

import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np


# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, parent_dir)
from muscle_redundancy_solver import analyse_mrs_results

# path information
main_datapath = 'E:\\Data\\Wang2023\\Processed_data\\Processed_data'
osim_model_path = os.path.join(main_datapath,'Subj04','OS','UTmodel',
                               'gait2392_simbody_subj04_scaled.osim')
results_folder = os.path.join(main_datapath,'Subj04','OS','mrs_TGCS')

#---------------------------------------------------------------
# ----------------- Notes                      -----------------
#---------------------------------------------------------------

# er is duidelijk nog iets mis met de optimalisaties met de linear actuator
# de activaties van de actuator zijn veel te hoog. er is ook heel veel co-activatie

# ik vertrouw data van Wang precies niet helemaal. residuals zijn heel hoog. veel ruis op de data...
# toch eens kijken naar data van Tim Van Der Zee ?
# (moet ik eerst exporteren van van addbiomechanics via ubuntu laptop)
# eerst gewoon cardiff data gebruiken. Die vertrouw ik en lukt op 1 snelheid. andere
# snelheden volgen nog in de toekomst



#---------------------------------------------------------------
# ----------------- test on slow walking speed -----------------
#---------------------------------------------------------------
files = ['Subj04walk_18_linact_minact.pkl',
         'Subj04walk_18_linact_minP.pkl',
         'Subj04walk_18_mrs_minact.pkl',
         'Subj04walk_18_mrs_minP.pkl']

file_header = ['linact minact', 'linact minP', 'mrs minact', 'mrs minP']

total_net_work = []
total_pos_work = []
total_neg_work = []


for ifile in range(len(files)):
    # import data
    filename = os.path.join(results_folder, files[ifile])
    # Load the pickle file
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    # default analysis simulation results
    data_analysis = analyse_mrs_results(data)
    # get total mechanical work by muscle fibers
    total_net_work.append(np.sum(data_analysis['fiberwork']))
    total_pos_work.append(np.sum(data_analysis['fiberwork_pos']))
    total_neg_work.append(np.sum(data_analysis['fiberwork_neg']))
    # make file with power of muscles
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data['t'], data['fiber_power'].T)
    plt.subplot(2, 1, 2)
    plt.plot(data['t'], data['a'].T)

# Create bar plots
fig, axs = plt.subplots(3, 1)

# Plot total_net_work
axs[0].bar(files, total_net_work, color='b')
axs[0].set_title('Total Net Work')
axs[0].set_ylabel('Work (J)')
axs[0].set_xticklabels(file_header, rotation=0, ha='right')

# Plot total_pos_work
axs[1].bar(files, total_pos_work, color='g')
axs[1].set_title('Total Positive Work')
axs[1].set_ylabel('Work (J)')
axs[1].set_xticklabels(file_header, rotation=0, ha='right')

# Plot total_neg_work
axs[2].bar(files, total_neg_work, color='r')
axs[2].set_title('Total Negative Work')
axs[2].set_ylabel('Work (J)')
axs[2].set_xticklabels(file_header, rotation=0, ha='right')

# Adjust layout
plt.tight_layout()


#---------------------------------------------------------------------------------
# ----------------- test mrs and linactutor min act on walk speeds ---------------
#---------------------------------------------------------------------------------

files = ['Subj04walk_54_linact_minact.pkl',
         'Subj04walk_54_mrs_minact.pkl',
         'Subj04walk_45_linact_minact.pkl',
         'Subj04walk_45_mrs_minact.pkl',
         'Subj04walk_36_linact_minact.pkl',
         'Subj04walk_36_mrs_minact.pkl']

file_header = ['linact 56', 'mrs 56', 'linact 45', 'mrs 45', 'linact 36', 'mrs 36']

total_net_work = []
total_pos_work = []
total_neg_work = []


for ifile in range(len(files)):
    # import data
    filename = os.path.join(results_folder, files[ifile])
    # Load the pickle file
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    # default analysis simulation results
    data_analysis = analyse_mrs_results(data)
    # get total mechanical work by muscle fibers
    total_net_work.append(np.sum(data_analysis['fiberwork']))
    total_pos_work.append(np.sum(data_analysis['fiberwork_pos']))
    total_neg_work.append(np.sum(data_analysis['fiberwork_neg']))
    # make file with power of muscles
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data['t'], data['fiber_power'].T)
    plt.subplot(2, 1, 2)
    plt.plot(data['t'], data['a'].T)



# Create bar plots
fig, axs = plt.subplots(3, 1)

# Plot total_net_work
axs[0].bar(files, total_net_work, color='b')
axs[0].set_title('Total Net Work')
axs[0].set_ylabel('Work (J)')
axs[0].set_xticklabels(file_header, rotation=0, ha='right')

# Plot total_pos_work
axs[1].bar(files, total_pos_work, color='g')
axs[1].set_title('Total Positive Work')
axs[1].set_ylabel('Work (J)')
axs[1].set_xticklabels(file_header, rotation=0, ha='right')

# Plot total_neg_work
axs[2].bar(files, total_neg_work, color='r')
axs[2].set_title('Total Negative Work')
axs[2].set_ylabel('Work (J)')
axs[2].set_xticklabels(file_header, rotation=0, ha='right')

# Adjust layout
plt.tight_layout()


# ---------------------------------------------------------------------------------
# ----------------- Debugging   -------------------
# ---------------------------------------------------------------------------------

filename = os.path.join(results_folder, 'Subj04walk_54_linact_minact.pkl')

# make file of inverse dynamic moment as a function of time
filename = os.path.join(results_folder, files[ifile])
# Load the pickle file
with open(filename, 'rb') as file:
    solution = pickle.load(file)

plt.figure()
# loop over dofs
dofs = ['knee_angle_r', 'ankle_angle_r', 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r']
for dof in range(len(dofs)):
    plt.subplot(len(dofs), 1, dof + 1)
    plt.plot(solution['t'], solution['tau_ideal'][dof, :].T, label='ideal actuator')
    plt.plot(solution['t'], solution['muscle_torque'][dof, :].T, label='muscle torque')
    plt.plot(solution['t'], solution['id'][dof, :].T, label='inverse dynamics')
    plt.xlabel('time [s]')
    plt.ylabel('torque [Nm]')
    plt.title(dofs[dof] + ' torque')
plt.legend()












plt.show()

# for ifile in range(len(files)):
#     # import data
#     filename = os.path.join(results_folder, files[ifile])
#     # Load the pickle file
#     with open(filename, 'rb') as file:
#         data = pickle.load(file)
#     # default analysis simulation results
#     data_analysis = analyse_mrs_results(data)
#     # get total mechanical work by muscle fibers
#     total_net_work = np.sum(data_analysis['muscle_work'])
#     total_pos_work = np.sum(data_analysis['muscle_pos_work'])
#     total_neg_work = np.sum(data_analysis['muscle_neg_work'])
#
#



