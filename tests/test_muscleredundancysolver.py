# simple script to test the muscle redundancy solver
#-----------------------------------------------------

import os
from muscle_redundancy_solver import muscle_redundancy_solver



# path to data
mainpath = 'C:/Users/mat950/Documents/Software/general_tools/python_toolkit/osim_tools_python'
osim_model_path = os.path.join(mainpath,'data','subject1.osim')
ikfile = os.path.join(mainpath,'data','Walking_IK.mot')
idfile = os.path.join(mainpath,'data','Walking_ID.sto')


# create muscle redundancy solver object
my_mrs = muscle_redundancy_solver(osim_model_path, ikfile, idfile)

# set dofs
dofs = ['hip_flexion_r','knee_angle_r','ankle_angle_r']
my_mrs.set_dofs(dofs)

# identify muscles for selected dofs
muscles_sel = my_mrs.identify_muscles()
print(muscles_sel)

