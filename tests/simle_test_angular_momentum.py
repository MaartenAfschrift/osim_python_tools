

# import utilities
from osim_utilities import osim_subject
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

import opensim as osim
from pathlib import Path
import os
import numpy as np
import pandas as pd
from general_utilities import readMotionFile
from scipy.interpolate import UnivariateSpline
from kinematic_analyses import bodykinematics

matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures



# path to data
datapath = 'C:/Users/mat950/Documents/Data/DebugOsimCode'

# init an opensim subject
#osim_model = os.path.join(datapath, 'subject01.osim')
#ikfile = os.path.join(datapath,'subject01_walk1_ik.mot')
osim_model = os.path.join(datapath, 'OsimModel.osim')
ikfile = os.path.join(datapath,'KS_Refwalk.mot')
ik = readMotionFile(ikfile)

# low pass filter ik data (ik is a pandas dataframe)
model = osim.Model(osim_model)
state= model.initSystem()

# Initialize a DataFrame to store the derivatives
ik_dot = pd.DataFrame()
ik_dot['time'] = ik['time']

# Compute the time derivatives using spline interpolation
for col in ik.columns:
    if col != 'time':
        spline = UnivariateSpline(ik['time'], ik[col], s=0)
        ik_dot[col] = spline.derivative()(ik['time'])

coordset = model.getCoordinateSet()
ncoords = coordset.getSize()

L = np.zeros((len(ik['time']), 3))
com = np.zeros((len(ik['time']), 3))
comd = np.zeros((len(ik['time']), 3))
# this implementation with settings states based on ik file is a lot easier and should also be implemented
# in the methods to compute mucsle tendon lengths and moment arms
for t in range(0, len(ik.time)):
    for i in range(0, ncoords):
        name = coordset.get(i).getName()
        if (coordset.get(i).getMotionType() == 1):
            # rotational dof
            coordset.get(name).setValue(state, ik[name][t]*np.pi/180)
            coordset.get(name).setSpeedValue(state, ik_dot[name][t]*np.pi/180)
        else:
            # translation dof
            coordset.get(name).setValue(state, ik[name][t])
            coordset.get(name).setSpeedValue(state, ik_dot[name][t])

    # get whole body angular momentum
    model.realizeVelocity(state)
    L_vec3 = model.calcAngularMomentum(state)
    L[t, 0]  = L_vec3.get(0)
    L[t, 1] = L_vec3.get(1)
    L[t, 2] = L_vec3.get(2)
    r_com = model.calcMassCenterPosition(state)
    com[t, 0] = r_com.get(0)
    com[t, 1] = r_com.get(1)
    com[t, 2] = r_com.get(2)
    v_com = model.calcMassCenterVelocity(state)
    comd[t, 0] = v_com.get(0)
    comd[t, 1] = v_com.get(1)
    comd[t, 2] = v_com.get(2)


# also use default bodykinematics function to double check com computation
#bk = bodykinematics(osim_model, datapath, ikfile)
bk_pos_api = readMotionFile(os.path.join(datapath, 'KS_Refwalk_BodyKinematics_pos_global.sto'))
bk_vel_api = readMotionFile(os.path.join(datapath, 'KS_Refwalk_BodyKinematics_vel_global.sto'))

# check if COM state is the same
plt.figure()
for i in range(0,3):
    plt.subplot(1,3,i+1)
    plt.plot(ik.time, com[:,i])
    if i == 0:
        plt.plot(bk_pos_api.time, bk_pos_api.center_of_mass_X, '--')
    if i == 1:
        plt.plot(bk_pos_api.time, bk_pos_api.center_of_mass_Y, '--')
    if i == 2:
        plt.plot(bk_pos_api.time, bk_pos_api.center_of_mass_Z, '--')

plt.legend(['model','analysis'])


plt.figure()
for i in range(0,3):
    plt.subplot(1,3,i+1)
    plt.plot(ik.time, comd[:,i])
    if i == 0:
        plt.plot(bk_vel_api.time, bk_vel_api.center_of_mass_X, '--')
    if i == 1:
        plt.plot(bk_vel_api.time, bk_vel_api.center_of_mass_Y, '--')
    if i == 2:
        plt.plot(bk_vel_api.time, bk_vel_api.center_of_mass_Z, '--')

plt.legend(['model','analysis'])

#norm_factor = model.getTotalMass(state)
# seems to give same results as Herr and Todorov paper. so seems to work
plt.figure()
plt.plot(ik.time,L)
plt.legend(['x','y','z'])
plt.show()