
import numpy as np
import matplotlib.pyplot as plt

# static methods for this class
def force_velocity(norm_fiber_velocity):
    # Equation S4:
    d1 = -0.318
    d2 = -8.149
    d3 = -0.374
    d4 = 0.886
    velocity_force = d1 * np.log(d2 * norm_fiber_velocity + d3 +
                                 np.sqrt((d2 * norm_fiber_velocity + d3) ** 2 + 1)) + d4

    return velocity_force


def force_length_fiber(norm_fiber_length):
    b1i = np.array([0.815, 0.433, 0.100])
    b2i = np.array([1.055, 0.717, 1.000])
    b3i = np.array([0.162, -0.030, 0.354])
    b4i = np.array([0.063, 0.200, 0.000])

    fact = 0

    for ii in range(3):
        fact_tmp = b1i[ii] * np.exp((-0.5 * (norm_fiber_length - b2i[ii]) ** 2) /
                                    (b3i[ii] + b4i[ii] * norm_fiber_length) ** 2)
        fact += fact_tmp
    return (fact)


def inverse_force_velocity(velocity_force):
    # Equation S13:
    d1 = -0.318
    d2 = -8.149
    d3 = -0.374
    d4 = 0.886
    norm_fiber_velocity = (np.sinh((velocity_force - d4) / d1) - d3) / d2
    return norm_fiber_velocity


def force_length_tendon(norm_tendon_length, kT, tendon_shift):
    # 'evaluating eqs. S18, 3, and S19'
    c1 = 0.200
    c2 = 0.995
    c3 = 0.250
    # Equation S1:
    ft = c1 * np.exp(kT * (norm_tendon_length - c2)) - c3 + tendon_shift
    return (ft)


def inverse_force_length_tendon(ft, kT, tendon_shift):
    # 'evaluating eqs. S18, 3, and S19'
    c1 = 0.200
    c2 = 0.995
    c3 = 0.250
    # Equation S1:
    norm_tendon_length = np.log((ft + c3 - tendon_shift) / c1) / kT + c2
    return norm_tendon_length


def default_force_length_par_elastic(norm_fiber_length):
    kpe = 4.0
    e0 = 0.6
    t5 = np.exp(kpe * (norm_fiber_length - 0.10e1) / e0)
    passive_fiber_force = ((t5 - 0.10e1) + 0.995172050006169) / 53.598150033144236
    return (passive_fiber_force)


# model properties
q_dotmax = 10
qmin = 0
qmax = 3.5
qopt = -61/180 * np.pi  # optimal fiber length in radians
qrange = 61/180* np.pi*1.3 # qrange radians is equal to optimal fiber length (improve explanation here)

# kracht leng zou ik jackson nemen
# kracht snelheid zou zie force_velocity



# sample data
qdot_sample  = np.linspace(-q_dotmax, q_dotmax, 100)
qsample = np.linspace(-q_dotmax, q_dotmax, 100)


#inputs for force-length and velocity
qdot_norm = qdot_sample/q_dotmax
qnorm = ((qsample - qopt)/qrange+1)


# evaluate model
fv = force_velocity(qdot_norm)
fl = force_length_fiber(qnorm)
plt.figure()
plt.plot(qdot_sample, fv)
plt.plot(qsample , fl)
plt.show()
