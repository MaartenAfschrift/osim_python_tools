import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
g = 9.81  # Gravity (m/s^2)
L = 1.0  # Leg length (m)
m = 1.0  # Leg mass (kg)
gamma = 0.05  # Slope angle (radians)


# Equations of motion
def gait_model(t, y):
    theta, phi, dtheta, dphi = y  # Unpack state variables
    # Equations of motion
    ddtheta = (
                      -g * np.sin(theta - gamma)
                      - (dphi ** 2) * np.sin(theta - phi) * np.cos(theta - phi)
              ) / (L * (1 + np.cos(theta - phi) ** 2))

    ddphi = (
                    -g * np.sin(phi - gamma)
                    + (dtheta ** 2) * np.sin(theta - phi) * np.cos(theta - phi)
            ) / (L * (1 + np.cos(theta - phi) ** 2))

    return [dtheta, dphi, ddtheta, ddphi]


# Collision dynamics
def collision(y):
    theta, phi, dtheta, dphi = y
    # Swap legs and apply energy loss
    theta_new = phi
    phi_new = theta
    dtheta_new = dphi * (1 - np.cos(theta - phi))
    dphi_new = dtheta * (1 - np.cos(theta - phi))
    return [theta_new, phi_new, dtheta_new, dphi_new]


# Simulation
def simulate(initial_conditions, t_max, dt):
    t = 0
    y = initial_conditions
    trajectory = [y]
    time = [t]

    while t < t_max:
        # Integrate equations of motion until the swing leg hits the ground
        sol = solve_ivp(gait_model, [t, t + dt], y, t_eval=[t + dt])
        y = sol.y[:, -1]
        t += dt

        # Check for collision (swing leg angle passes through zero)
        if y[1] <= 0:
            y = collision(y)

        trajectory.append(y)
        time.append(t)

    return np.array(time), np.array(trajectory)


# Initial conditions: [theta, phi, dtheta, dphi]
initial_conditions = [0.2, -0.2, 0, 0]

# Simulate for 10 seconds
t_max = 10
dt = 0.01
time, trajectory = simulate(initial_conditions, t_max, dt)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, trajectory[:, 0], label='Theta (Stance leg)')
plt.plot(time, trajectory[:, 1], label='Phi (Swing leg)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.title('Passive Dynamic Walking Simulation')
plt.grid()
plt.show()