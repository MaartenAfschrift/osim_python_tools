# simple test for debuggin code in surrogate_muscle_tendon_geom model

# add path above this folder to the search directory so that I can acces osim_utilities.py
from pathlib import Path
import sys
import os
mainfolder = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, mainfolder)

# import opensim utilities
from osim_utilities import osim_subject



# open a model
parent_path = Path(__file__).resolve().parent.parent
model_path = os.path.join(parent_path,'data','subject1.osim')
my_subject = osim_subject()

# get coordinates of this model
coordinates = my_subject.coord_names
muscles = my_subject.muscle_names


print(coordinates)
print(muscles)

print('test')
# get muscles in this model

