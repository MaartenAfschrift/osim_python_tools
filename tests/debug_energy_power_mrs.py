import os
import pickle
import sys
from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
#from general_utilities import readMotionFile
#from mocap_utilities import detect_heelstrike_toeoff

import matplotlib
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures
import pandas as pd

out_path_main = 'C:\\Users\\\mat950\\\Documents\\\Data\\\Sim_TGCS\\\CardiffData'
subject =  'Patient10'
results_folder = os.path.join(out_path_main, subject + '_test_energy')
file = 'Refwalk_linact_minact_4.pkl'
filename = os.path.join(results_folder, file)
with open(filename, 'rb') as file:
    data = pickle.load(file)


# check energy equations
plt.figure()
plt.plot(data['t'], data['joint_power'].T, label ='joint power')
plt.plot(data['t'], np.sum(data['muscle_power'], axis=0).T, label = 'muscle power')
plt.plot(data['t'], np.sum(data['power_actuator'], axis=0).T, label = 'ideal actuator power')
plt.plot(data['t'], np.sum(data['fiber_power'], axis=0).T, label = 'fiber power')
plt.plot(data['t'], np.sum(data['tendon_power'], axis=0).T, label = 'tendon power')
power_muscle = data['tendon_power'] + data['fiber_power']
plt.plot(data['t'], np.sum(power_muscle, axis=0).T, label = 'sum fiber tendon', linestyle='--') # I want a dotted line here

plt.legend()
plt.xlabel('time [s]')
plt.ylabel('power [W]')


plt.figure()
plt.plot(data['t'], data['muscle_torque'].T)
plt.plot(data['t'], data['id'].T)

plt.figure()
plt.plot(data['ik'][0,:], data['lmt'][0,:],linestyle='', marker='o')  # Pl)



# Note: we have to first filter IK and subsequently compute the muscle-tendon length







# print all headers
for key in data.keys():
    print(key)

plt.show()




