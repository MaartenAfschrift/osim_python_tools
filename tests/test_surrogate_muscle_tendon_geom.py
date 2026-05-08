# simple test for debuggin code in surrogate_muscle_tendon_geom model

# add path above this folder to the search directory so that I can acces osim_utilities.py
from pathlib import Path
import sys
import os
import numpy as np
#mainfolder = str(Path(__file__).resolve().parents[1])
# temp different
mainfolder = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, mainfolder)
print(mainfolder)

# import opensim utilities
from osim_utilities import osim_subject
from surrogate_muscle_tendon_geom import (
    create_dummy_motion,
    fit_polynomials_for_muscles,
    evaluate_muscle_polynomials,
    create_casadi_muscle_polynomial_function,
)



# open a model
model_path = os.path.join(mainfolder,'data','subject1.osim')
my_subject = osim_subject(model_path)

# get coordinates of this model
coordinates = my_subject.coord_names
muscles = my_subject.muscle_names

# coordinates I want to use for fitting, all other coordinates should be locker
# in their default position [I assume 0 for this model]
coordinates_fit = ['hip_flexion_r','knee_angle_r','ankle_angle_r']
lowerbound = [-70, -120, -40]
upperbound = [70, 0, 40]
# array with zeros the size of coordinates
default_coordinates = [0.0] * len(coordinates)
# other settings for creating the dummy motion
n_datapoints = 1000
ik_file = os.path.join(mainfolder,'tests','dummy_lhs_ik.mot')

# fit polynomials to muscles
muscle_poly_info = my_subject.fit_polynomial(coordinates_fit, lowerbound, upperbound,
                          n_datapoints=1000,
                          coordinate_defaults=default_coordinates)

# evaluate output
print("n muscles fitted:", len(muscle_poly_info["muscle"]))
for mdat in muscle_poly_info["muscle"][:10]:
    print(
        mdat["m_name"],
        "dof=", mdat["DOF"],
        "order=", mdat["order"],
        "lmt_rmse=", mdat["lMT_error_rms"],
    )

# Example: evaluate fitted polynomials for all dummy-motion samples.
lmt_pred, dm_pred = evaluate_muscle_polynomials(
    muscle_poly_info,
    q_data=my_subject.ikdat[0],
    q_in_degrees=True,
)

# example create a casadi function that takes as input the coordinates (1 time point) and
# outputs the muscle-tendon lengths and moment arms
casadi_fun = create_casadi_muscle_polynomial_function(muscle_poly_info)
q0_deg = my_subject.ikdat[0][muscle_poly_info["coordinates_fit"]].iloc[0].to_numpy(dtype=float)
q0_rad = q0_deg * np.pi / 180.0
lmt0, dm0 = casadi_fun(q0_rad)
print("casadi lMT shape:", lmt0.shape)
print("casadi dM shape:", dm0.shape)

print("predicted lMT columns:", list(lmt_pred.columns[:6]))
print("predicted dM columns:", list(dm_pred.columns[:6]))
