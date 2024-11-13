# osim_tools_python

contains several python tools for workflows in opensim. I typically include this repo as an submodule in my projects. Creating a python package would probably be even better, but I dont know how to do this and the code is currently not at that stage...

I stole some things from pyosim (InverseKinematics and InverseDynamics) and utilsTRC from OpenCap.

## workfow

The main worflow to process data assumes that you collected marker data (and potentially also ground reaction forces) and have already scaled your musculoskeletal model in opensim (.osim file). Scaling is intentionally not included in this processing pipeline as this step requires a manual check in my opinion.

You can use the class *osim_subject* to process your data. You can create an object of this class using.

```python
# create object for processing
subj = osim_subject(model_path)
```

You can assign a specific trc file or a folder with trc files to this osim_subject.

```python
subj.set_trcfiles_fromfolder(trc_folder)
```

Note that if you assign a folder, it will read all the .trc files in this folder and assign the different filenames as different 'trials'. All future analsys will be run on all trials. For example if you want to compute inverse kinematics:

```python
# set .xml file with ik settings (typically exported from GUI)
subj.set_general_ik_settings(general_ik_settings)
# set directory to save ik results
subj.set_ik_directory(ik_folder)
# compute inverse kinematics (here prevent overwriting existing files)
subj.compute_inverse_kinematics(overwrite = False)
```

Other analyses that have been implemented are:

**Bodykinematics**

```python
# run bodykinematics
subj.set_bodykin_folder(bk_folder)
subj.compute_bodykin(overwrite = False)
```

**Inverse dynamics**

Set the external loads

```python
# path to general settings for external loads (typically exported from GUI)
forces_settings = os.path.join(MainDatapath, 'settings', 'my_ext_loads.xml')  
subj.set_generic_external_loads(forces_settings)
# directory to save the external loads files
subj.set_ext_loads_dir(ext_loads_dir)
subj.create_extloads_files() # this should be an implicit function, adapt
```

Run inverse dynamics

```python
# generic id settings (typically exported from GUI)
subj.set_general_id_settings(general_id_settings)
subj.set_id_directory(my_id_dir)
subj.compute_inverse_dynamics(boolRead = True)
```

**Muscle analysis**

As the analysis can be pretty slow to compute muscle-tendon lengths and moments arms (as this tool always solves for equilibrium between tendon and muscle force) I also implemented a version using that API that only uses the generalized coordinates (this makes it faster, but the communication with API limits the speed-up).

```python
# compute muscle-tendon lengths and moment arms using api
subj.set_lmt_folder(lmt_folder)
subj.compute_lmt()
subj.set_momentarm_folder(dm_folder)
subj.compute_dM()
```

Note that implementations uses by default the fast version to compute the moment arms. In this fast version I first identify the dofs spanned by each muscle and only evaluate the moment arm for those dofs.

### DeGroote2016 muscle model

This repository also contains a class to compute muscle dynamics as in DeGroote2016. It is implemented with normalised fiber length as a state and you can use it in both implicit and explicit formulations. 

example to create muscle object

```python
FMo = 1000 # maximal isometric force in N
lMo = 0.2 # optimal fiber length in m
lTs = 0.5 # tendon slack length in m
alpha = 0.1 # optimal pennation angle in rad
vMtildemax = 10 # maximal muscle fiber velocity in lMo/s
kT = 35 # F/L propeties tendon
muscle = DeGrooteMuscle(FMo, lMo, lTs, alpha, vMtildemax, kT)
```

example explicit formulation contraction dynamics

```python
muscle.set_activation(0.1)
muscle.set_norm_fiber_length(1)
muscle.set_muscle_tendon_length(0.7)
lmtilde_dot = muscle.get_norm_fiber_length_dot()
```

example implicit formulation

```python
muscle.set_activation(0.1)
muscle.set_norm_fiber_length(1)
muscle.set_muscle_tendon_length(0.7)
muscle.set_norm_fiber_length_dot(lmtilde_dot)
#muscle.set_norm_fiber_velocity(lmtilde_dot/muscle.maximal_fiber_velocity)
hill_err = muscle.get_hill_equilibrium()
print(hill_err)
```


