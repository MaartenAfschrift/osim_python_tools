# analyse simulation results cardifdata


import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from general_utilities import readMotionFile
from mocap_utilities import detect_heelstrike_toeoff


import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures
import pandas as pd



# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, parent_dir)
from muscle_redundancy_solver import analyse_mrs_results

out_path_main = 'C:\\Users\\\mat950\\\Documents\\\Data\\\Sim_TGCS\\\CardiffData'
subject =  'Patient10'
results_folder = os.path.join(out_path_main, subject)


#---------------------------------------------------------------
# ----------------- test on slow walking speed -----------------
#---------------------------------------------------------------
files = ['Refwalk_linact_minact.pkl',
         'Refwalk_mrs_minact.pkl']

file_header = ['linact minact', 'mrs_minact']

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



# ---------------------------------------------------------------
# ----------------- Debugging -----------------------------------
# ---------------------------------------------------------------

out_path_main = 'C:\\Users\\\mat950\\\Documents\\\Data\\\Sim_TGCS\\\CardiffData'
subject =  'Patient10'
results_folder = os.path.join(out_path_main, subject + '_debug')


#---------------------------------------------------------------
# ----------------- test on slow walking speed -----------------
#---------------------------------------------------------------
files = ['Refwalk_mrs_minact_15.pkl',
         'Refwalk_linact_minact_15.pkl',
         'Refwalk_mrs_minPsq_1.pkl',
         'Refwalk_linact_minPsq_1.pkl']

file_header = ['mrs minact', 'linact minact', 'mrs minP', 'linact minP']
a_sq = []
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
    # compute asquared
    a =  data['a']
    a_sq.append(np.sum(a[:]**2))

    total_net_work.append(np.sum(data_analysis['fiberwork']))
    total_pos_work.append(np.sum(data_analysis['fiberwork_pos']))
    total_neg_work.append(np.sum(data_analysis['fiberwork_neg']))


    # make file with power of muscles
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(data['t'], data['fiber_power'].T)
    plt.subplot(3, 1, 2)
    plt.plot(data['t'], data['a'].T)
    plt.subplot(3, 1, 3)
    plt.plot(data['t'], data['tendon_power'].T)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(data['t'], data['tendon_force'].T)
    plt.subplot(3, 1, 2)
    plt.plot(data['t'], data['tendon_velocity'].T)
    plt.subplot(3, 1, 3)
    plt.plot(data['t'], data['fiber_velocity'].T)


    print('a squared: ', np.sum(a[:]**2))

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

# ---------------------------------------------------------------
# extensive validation based on detected gait events
# ---------------------------------------------------------------
main_datapath = 'C:\\Users\\mat950\\Documents\\Data\\Cardiff_Lonit\\osimData_young'
grf_file = os.path.join(main_datapath, subject, 'RefWalk', 'data','Refwalk_GRF.mot')
grf_data = readMotionFile(grf_file)

# detect gait cycle event based on vertical ground reaction force
Fy_L = grf_data['ground_force_vy']
Fy_R = grf_data['1_ground_force_vy']

plt.figure()
plt.plot(grf_data['time'], Fy_L, label='Fy_L')
plt.plot(grf_data['time'], Fy_R, label='Fy_R')

time = grf_data.time
[ths, tto, hs, to] =  detect_heelstrike_toeoff(time, Fy_R, 50, dtOffPlate = 0.3)

files = ['Refwalk_linact_minact.pkl',
         'Refwalk_linact_minact_opt_tendon.pkl',
         'Refwalk_linact_minPsq.pkl',
         'Refwalk_linact_minPsq_opt_tendon.pkl',
         'Refwalk_mrs_minact.pkl',
         'Refwalk_mrs_minact_opt_tendon.pkl',
         'Refwalk_mrs_minPsq.pkl',
         'Refwalk_mrs_minPsq_opt_tendon.pkl']

#results_folder = os.path.join(out_path_main, subject + '_1403_v4')
results_folder = os.path.join(out_path_main, subject + '_2803')
# compute mechanical work for each gait cycle (between two heel strikes)
results = {}

for ifile in range(len(files)):
    # import data
    filename = os.path.join(results_folder, files[ifile])
    # Load the pickle file
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    # get start time datafile
    t0 = data['t'][0]
    tend = data['t'][-1]
    # find all heel strikes between t0 and tend
    isel = np.where((ths > t0) & (ths < tend))[0]
    # init variables
    total_net_work = []
    total_pos_work = []
    total_neg_work = []
    asq_mean = []
    for i in range(len(isel)-1):
        tstart_gc = ths.iloc[isel[i]]
        tend_gc = ths.iloc[isel[i+1]]

        # default analysis simulation results
        data_analysis = analyse_mrs_results(data, tstart=tstart_gc, tend=tend_gc)

        total_net_work.append(np.sum(data_analysis['fiberwork']))
        total_pos_work.append(np.sum(data_analysis['fiberwork_pos']))
        total_neg_work.append(np.sum(data_analysis['fiberwork_neg']))
        # total_net_work.append(np.sum(data_analysis['tendonwork']))
        # total_pos_work.append(np.sum(data_analysis['tendonwork_pos']))
        # total_neg_work.append(np.sum(data_analysis['tendonwork_neg']))
        # total_net_work.append(np.sum(data_analysis['musclework']))
        # total_pos_work.append(np.sum(data_analysis['musclework_pos']))
        # total_neg_work.append(np.sum(data_analysis['musclework_neg']))
        asq_mean.append(np.sum(data_analysis['asq_mean']))


    # store outputs
    results[files[ifile][0:-4]] = {'net_work': total_net_work,
                                   'pos_work': total_pos_work,
                                   'neg_work': total_neg_work,
                                   'asq_mean': asq_mean}

# Create plot
plt.figure()
for ifile in range(len(files)):
    plt.subplot(2,2,1)
    fileinfo = files[ifile][0:-4]
    plt.plot(results[fileinfo]['net_work'], label=files[ifile][0:-4])
    # plt.xlabel('Gait cycle')
    plt.title('Total Net Work')

    plt.subplot(2,2,2)
    plt.plot(results[fileinfo]['pos_work'], label=files[ifile][0:-4])
    # plt.xlabel('Gait cycle')
    plt.title('Total Positive Work')

    plt.subplot(2,2,3)
    plt.plot(results[fileinfo]['neg_work'], label=files[ifile][0:-4])
    plt.xlabel('Gait cycle')
    plt.title('Total Negative Work')

    plt.subplot(2,2,4)
    plt.plot(results[fileinfo]['asq_mean'], label=files[ifile][0:-4])
    plt.xlabel('Gait cycle')
    plt.title('Mean Activation Squared')

plt.legend()

plt.show()