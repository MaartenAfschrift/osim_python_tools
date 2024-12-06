# test if chatgpt can do this ("if students can do this without thinking")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sympy as sp



# Parameters for the simulation
g = 9.81  # gravity (m/s^2)
L = 1.0  # leg length (m)
mass = 70
dt = 0.001  # time step (s)
J_Q = mass*L**2
theta_max = 0.6  # maximum angle from vertical (rad)
theta_min = -0.6  # minimum angle for next step transition (rad)

# Initial conditions
theta = theta_max  # initial pendulum angle (rad)
omega = -2  # initial angular velocity (rad/s)

# symbolic equation for plastic collision with ground
# Define symbols
fi, fi_post, fid, fid_post, L, mass = sp.symbols('fi fi_post fid fid_post L mass', real=True)
r = sp.Matrix([L * sp.cos(fi), L * sp.sin(fi), 0])
r_post = sp.Matrix([L * sp.cos(fi_post), L * sp.sin(fi_post), 0])
v = sp.Matrix([-L * fid * sp.sin(fi), L * fid * sp.cos(fi), 0])
v_post = sp.Matrix([-L * fid_post * sp.sin(fi_post), L * fid_post * sp.cos(fi_post), 0])
eq_Ang = sp.Eq(r.cross(mass * v)[2] - r_post.cross(mass * v_post)[2],0)  # Use .cross()
solution = sp.solve(eq_Ang, fid_post)
# this shows that fid = fid_post (which makes sense obviously)
print(solution)

# Time parameters
T_total = 5.0  # total simulation time (s)
time = np.arange(0, T_total, dt)

# Arrays to store results
theta_values = []
omega_values = []
time_values = []


# Simulation loop
for t in time:
    # express the pendulum kinematics in standard xy coordinate system
    fi = theta + np.pi/2

    # Calculate angular acceleration (pendulum dynamics)
    G = np.array([0, -mass*g])
    r = L * np.array([np.cos(fi), np.sin(fi)])
    MG = np.cross(r, G)
    alpha = MG/J_Q

    # Check for step transition
    # this is currently wrong in my opinion. As this is a plastic collision with the ground it
    # should be that velocity contact point is zero and angular momentum w.r.t. point should be maintained
    # this will result in energy dissipation and hence some energy input is needed
    if theta + omega * dt < theta_min:
        theta = theta_max  # reset pendulum angle for the next step
        v = L * omega * np.array([-np.sin(fi), np.cos(fi)])
        L_pre = np.cross(r, v)
        fi_post = theta + np.pi/2
        r_post = L * np.array([np.cos(fi_post), np.sin(fi_post)])
        L_pre - ()
        #


        #omega = omega  # carry over angular velocity to simulate walking

    # Update angular velocity and position using Euler integration
    omega += alpha * dt
    theta += omega * dt

    # Store values
    theta_values.append(theta)
    omega_values.append(omega)
    time_values.append(t)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(time_values, theta_values, label='Pendulum Angle (θ)')
plt.axhline(theta_max, color='r', linestyle='--', label='Step Reset (θ_max)')
plt.axhline(theta_min, color='g', linestyle='--', label='Step Transition (θ_min)')
plt.title('Inverted Pendulum Model of Walking')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid()

# Animation setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_title("Stick Figure Walking Animation")
ax.set(xlim=[-10, 1], ylim=[0, 2], xlabel='Time [s]', ylabel='Angle [rad]')

# Line to represent the data
line, = ax.plot([], [], 'b-', lw=2, label="Theta")  # Create a line for animation

def init():
    """Initialize the animation with an empty line."""
    line.set_data([], [])
    return line,

def update(frame):
    """Update the animation at each frame."""

    if not hasattr(update, "foot_hs_x"):
        update.foot_hs_x = 0  # Initialize on the first call

    if frame > 0:
        delta_theta = (theta_values[frame]-theta_values[frame-1])
        if delta_theta>0.2:
            update.foot_hs_x = update.foot_hs_x +  -2 * L * np.sin(theta_max)

    line.set_data([update.foot_hs_x, update.foot_hs_x+L*np.sin(theta_values[frame])],
                  [0, L*np.cos(theta_values[frame])])  # Update x and y data
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(time),
                              init_func=init, blit=True, interval=1)

# Show the animation
plt.legend()
plt.show()
