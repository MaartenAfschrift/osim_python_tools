# class with Friedl her muscle model

# L.M. Peeters & M. Afschrift

# This script defines the Hill muscle model class using the equations as proposed by De Groote et al. (2016)

import numpy as np
import matplotlib.pyplot as plt

# to do: add damping in parallel with the muscle in implicit implementation ? (not possible in explicit implementation)




class DeGrooteMuscle:

    # init class
    def __init__(self, FMo, lMo, lTs, alphao, vMtildemax, kT):

        # to do:
        #   1. change kT for strain at optimal fiber length ?

        # Define muscle properties
        self.maximal_isometric_force = FMo
        self.optimal_fiber_length = lMo
        self.tendon_slack_length = lTs
        self.optimal_pennation_angle = alphao
        self.maximal_fiber_velocity = vMtildemax
        self.kT = kT

        # Define parameters
        self.kpe = 4.0
        self.e0 = 0.6

        # Define instances
        self.mtLength = None
        self.activation = 0
        self.fiber_width = None
        self.tendon_length = None
        self.fiber_length = None
        self.norm_tendon_length = None
        self.tendon_force = None
        self.fiber_force = None
        self.norm_fiber_force = None
        self.passive_fiber_force = None
        self.active_fiber_force = None
        self.norm_fiber_velocity = None # [/lmtopt/vmax]
        self.norm_fiber_length_dot = None # time derivative of normalized fiber length [/lmopt]
        self.norm_fiber_length = None
        self.velocity_force = None
        self.active_fiber_force_denorm = None
        self.tendon_shift = 0


    ## Set functions
    def set_norm_fiber_length(self, lmtilde):
        self.norm_fiber_length = lmtilde

    def set_muscle_tendon_length(self, lmt):
        self.mtLength = lmt

    def set_activation(self, a):
        self.activation = a

    def set_norm_fiber_length_dot(self, lmtilde_dot):
        self.norm_fiber_velocity = lmtilde_dot / self.maximal_fiber_velocity

    def set_norm_fiber_velocity(self, vmtilde):
        self.norm_fiber_velocity = vmtilde

    # static methods for this class
    @staticmethod
    def force_velocity(norm_fiber_velocity):
        # Equation S4:
        d1 = -0.318
        d2 = -8.149
        d3 = -0.374
        d4 = 0.886
        velocity_force = d1 * np.log(d2 * norm_fiber_velocity + d3 +
                                     np.sqrt((d2 * norm_fiber_velocity + d3) ** 2 + 1)) + d4

        return velocity_force
    @staticmethod
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
        return(fact)


    @staticmethod
    def inverse_force_velocity(velocity_force):
        # Equation S13:
        d1 = -0.318
        d2 = -8.149
        d3 = -0.374
        d4 = 0.886
        norm_fiber_velocity = (np.sinh((velocity_force - d4) / d1) - d3) / d2
        return norm_fiber_velocity

    @staticmethod
    def force_length_tendon(norm_tendon_length, kT, tendon_shift):
        # 'evaluating eqs. S18, 3, and S19'
        c1 = 0.200
        c2 = 0.995
        c3 = 0.250
        # Equation S1:
        ft = c1 * np.exp(kT * (norm_tendon_length - c2)) - c3 + tendon_shift
        return(ft)

    @staticmethod
    def inverse_force_length_tendon(ft, kT, tendon_shift):
        # 'evaluating eqs. S18, 3, and S19'
        c1 = 0.200
        c2 = 0.995
        c3 = 0.250
        # Equation S1:
        norm_tendon_length = np.log((ft + c3 - tendon_shift) / c1) / kT + c2
        return norm_tendon_length


    ## Computations
    def compute_fiber_length(self):
        # 'denormalizing lMtilde'
        self.fiber_length = self.norm_fiber_length * self.optimal_fiber_length

    def compute_tendon_length(self):
        self.compute_fiber_length()
        # Based on equation S23:
        self.fiber_width = self.optimal_fiber_length * np.sin(self.optimal_pennation_angle)
        self.tendon_length = self.mtLength - np.sqrt((self.fiber_length ** 2 - self.fiber_width ** 2))

    def compute_norm_tendon_length(self):
        self.compute_tendon_length()

        # 'normalizing lT'
        self.norm_tendon_length = self.tendon_length / self.tendon_slack_length

    def compute_tendon_shift(self):
        generic_tendon_compliance = 35
        generic_tendon_shift = 0
        reference_norm_tendon_length = 1
        
        # compute tendon force at reference length
        reference_norm_tendon_force = (0.2 * np.exp(
            generic_tendon_compliance * (reference_norm_tendon_length - 0.995))
                                    - 0.25 + generic_tendon_shift)
        
        # compute tendon force with adjusted stiffness
        adjusted_norm_tendon_force = (0.2 * np.exp(
            self.kT * (reference_norm_tendon_length - 0.995))
                                   - 0.25 + generic_tendon_shift)

        # compute shift to have equal force at reference length
        self.tendon_shift = reference_norm_tendon_force - adjusted_norm_tendon_force

    def compute_tendon_force(self):
        # compute norm tendon length based on lmt and fiber length
        self.compute_norm_tendon_length()
        # computes shift in force-length curve tendon
        self.compute_tendon_shift()
        # computes norm tendon force
        ft = self.force_length_tendon(self.norm_tendon_length, self.kT, self.tendon_shift)
        # computes tendon force in N
        self.tendon_force = self.maximal_isometric_force * ft

    def compute_fiber_force(self):
        self.compute_fiber_length()
        self.compute_tendon_length()
        self.compute_tendon_force()

        # 'evaluating eqs. S18, 3, and S19
        # Equation S18:
        cos_alpha = (self.mtLength - self.tendon_length) / self.fiber_length
        # Equation S19
        self.fiber_force = self.tendon_force / cos_alpha
        return self.fiber_force

    def compute_norm_fiber_force(self):
        self.compute_fiber_force()

        # 'normalizing FM'
        self.norm_fiber_force = self.fiber_force / self.maximal_isometric_force

    def compute_passive_fiber_force(self):
        # Equation S3:
        t5 = np.exp(self.kpe * (self.norm_fiber_length - 0.10e1) / self.e0)
        self.passive_fiber_force = ((t5 - 0.10e1) + 0.995172050006169) / 53.598150033144236

    def compute_force_length(self):
        fact = self.force_length_fiber(self.norm_fiber_length)
        self.active_fiber_force = fact
        self.active_fiber_force_denorm = fact * self.activation * self.maximal_isometric_force
        return self.active_fiber_force, self.active_fiber_force_denorm

    def compute_norm_fiber_velocity(self):
        self.compute_passive_fiber_force()
        self.compute_force_length()
        self.compute_norm_fiber_force()

        # 'evaluating eqs. S14, S13, and S21'
        # Equation S14:
        fm_vtilde = ((self.norm_fiber_force - self.passive_fiber_force) /
                     (self.activation * self.active_fiber_force))
        self.norm_fiber_velocity = self.inverse_force_velocity(fm_vtilde)

    ## Implicit formulation (formulation 4 of DeGroote2016)
    def compute_velocity_force(self):
        # Equation S4:
        self.velocity_force = self.force_velocity(self.norm_fiber_velocity)
        return self.velocity_force

    def compute_velocity_force_linear_fv(self):
        # Linear relationship
        lin_a = 1
        lin_b = 1
        self.velocity_force = lin_a * self.norm_fiber_velocity + lin_b
        return self.velocity_force



    #---------------------------------
    #       Evaluate muscle dynamics
    #---------------------------------


    def compute_norm_fiber_length_dot(self):
        self.compute_norm_fiber_velocity()
        # Equation S21:
        self.norm_fiber_length_dot = self.maximal_fiber_velocity * self.norm_fiber_velocity
        return self.norm_fiber_length_dot


    def compute_hill_equilibrium(self):
        self.compute_fiber_length() # denorm fiber length
        self.compute_tendon_length() # tendon length from lmt and fiber length
        self.compute_force_length() # F/l relation
        self.compute_velocity_force() # F/v relation
        self.compute_passive_fiber_force() # passive F/l relation
        self.compute_tendon_force() # tendon force from tendon length

        # Equation S18:
        self.cosalpha = (self.mtLength - self.tendon_length) / self.fiber_length

        # Equation S31: equilibrium force tendon and force fiber + force passive
        hill_equilibrium = self.maximal_isometric_force * self.cosalpha * \
                          (self.activation * self.active_fiber_force * self.velocity_force + self.passive_fiber_force) \
                          - self.tendon_force

        return hill_equilibrium

    def compute_hill_equilibrium_linear_fv(self):
        self.compute_fiber_length() # denorm fiber length
        self.compute_tendon_length() # tendon length from lmt and fiber length
        self.compute_force_length() # F/l relation
        self.compute_velocity_force_linear_fv() # F/v relation
        self.compute_passive_fiber_force() # passive F/l relation
        self.compute_tendon_force() # tendon force from tendon length

        # Equation S18:
        self.cos_alpha = (self.mtLength - self.tendon_length) / self.fiber_length

        # Equation S31:
        hill_equilibrium = self.maximal_isometric_force * self.cos_alpha * \
                          (self.activation * self.active_fiber_force * self.velocity_force + self.passive_fiber_force) \
                          - self.tendon_force
        return hill_equilibrium

    def compute_muscle_force_rigid_tendon(self, lMt, vMt, activation):
        # fiber length and velocity are inputs
        lm_x = lMt- self.tendon_slack_length
        fiber_width = self.optimal_fiber_length * np.sin(self.optimal_pennation_angle)
        fiber_length = np.sqrt(lm_x ** 2 + fiber_width ** 2)
        self.norm_fiber_length = fiber_length / self.optimal_fiber_length
        self.norm_fiber_velocity = vMt / self.maximal_fiber_velocity / self.optimal_fiber_length
        self.compute_force_length() # F/l relation muscle fiber
        self.compute_velocity_force() # F/v relation muscle fiber
        self.compute_passive_fiber_force() # passive F/l relation
        # compute muscle force
        cosalpha = lm_x / fiber_length
        Fmuscle = (self.maximal_isometric_force * cosalpha *
                   (activation * self.active_fiber_force * self.velocity_force + self.passive_fiber_force))
        return(Fmuscle)


    #---------------------------------
    #       Plot functionalities
    #---------------------------------

    # plot force length and force velocity curve
    def plot_fl_curve(self):
        lmtilde_vect = np.linspace(0.2, 1.7, 100)
        fact_vect = np.zeros((len(lmtilde_vect), 1))
        fpass_vect = np.zeros((len(lmtilde_vect), 1))
        ct = 0
        for lmtilde in lmtilde_vect:
            self.set_norm_fiber_length(lmtilde)
            self.compute_passive_fiber_force()
            self.compute_force_length()
            fpass_vect[ct] = self.passive_fiber_force
            fact_vect[ct] = self.active_fiber_force
            ct = ct + 1
        plt.figure()
        plt.plot(lmtilde_vect*self.optimal_fiber_length,
                 fact_vect*self.maximal_isometric_force, label='Active Force')
        plt.plot(lmtilde_vect*self.optimal_fiber_length,
                 fpass_vect * self.maximal_isometric_force, label='Passive Force')
        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()
        plt.axvline(x=self.optimal_fiber_length, ymin=ymin,
                    ymax=ymax, color='k', linestyle='--')
        plt.axhline(y=self.maximal_isometric_force, xmin=xmin, xmax=xmax*10,
                    color='k', linestyle='--')
        plt.legend()
        plt.xlabel('Fiber Length [m]')
        plt.ylabel('Force [N]')


    def plot_fv_curve(self):
        vmtilde_vect = np.linspace(-1, 1, 100)
        fact_vect = np.zeros((len(vmtilde_vect), 1))
        ct = 0
        for vmtilde in vmtilde_vect:
            self.set_norm_fiber_velocity(vmtilde)
            fact_vect[ct] = self.compute_velocity_force()
            ct = ct + 1
        plt.figure()
        plt.plot(vmtilde_vect*self.maximal_fiber_velocity*self.optimal_fiber_length,
                 fact_vect * self.maximal_isometric_force)
        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()
        plt.axvline(x=0, ymin=ymin, ymax=ymax, color='k', linestyle='--')
        plt.axhline(y=self.maximal_isometric_force, xmin=xmin, xmax=xmax,
                    color='k', linestyle='--')

        plt.xlabel('Fiber Velocity [m/s]')
        plt.ylabel('Force [N]')

    def plot_fl_curve_norm(self):
        lmtilde_vect = np.linspace(0.2, 1.7, 100)
        fact_vect = np.zeros((len(lmtilde_vect), 1))
        fpass_vect = np.zeros((len(lmtilde_vect), 1))
        ct = 0
        for lmtilde in lmtilde_vect:
            self.set_norm_fiber_length(lmtilde)
            self.compute_passive_fiber_force()
            self.compute_force_length()
            fpass_vect[ct] = self.passive_fiber_force
            fact_vect[ct] = self.active_fiber_force
            ct = ct+1
        plt.figure()
        plt.plot(lmtilde_vect, fact_vect, label='Active Force')
        plt.plot(lmtilde_vect, fpass_vect, label='Passive Force')
        ymin, ymax = plt.ylim()
        plt.axvline(x=1, ymin=ymin, ymax=ymax, color='k', linestyle='--')
        plt.axhline(y=1, xmin = 0, xmax = 1.7,
                    color='k', linestyle='--')
        plt.legend()
        plt.xlabel('Normalized Fiber Length []')
        plt.ylabel('norm force []')


    def plot_fv_curve_norm(self):
        vmtilde_vect = np.linspace(-1, 1, 100)
        fact_vect = np.zeros((len(vmtilde_vect), 1))
        ct = 0
        for vmtilde in vmtilde_vect:
            self.set_norm_fiber_velocity(vmtilde)
            fact_vect[ct] = self.compute_velocity_force()
            ct = ct+1
        plt.figure()
        plt.plot(vmtilde_vect, fact_vect)
        ymin, ymax = plt.ylim()
        plt.axvline(x=0, ymin=ymin, ymax=ymax, color='k', linestyle='--')
        plt.axhline(y=1, xmin = np.min(vmtilde_vect), xmax = np.max(vmtilde_vect),
                    color='k', linestyle='--')

        plt.xlabel('Normalized Fiber Velocity []')
        plt.ylabel('norm force []')

    # get functions
    def get_norm_fiber_length(self):
        return self.norm_fiber_length

    def get_fiber_length(self):
        return self.norm_fiber_length * self.optimal_fiber_length

    def get_norm_fiber_velocity(self):
        return self.norm_fiber_velocity

    def get_fiber_velocity(self):
        return self.norm_fiber_length * self.optimal_fiber_length * self.maximal_fiber_velocity

    def get_tendon_shift(self):
        return self.tendon_shift

    def get_kT(self):
        return self.kT

    def get_maximal_isometric_force(self):
        return self.maximal_isometric_force

    def get_optimal_fiber_length(self):
        return self.optimal_fiber_length

    def get_force_length(self):
        return self.active_fiber_force

    def get_force_velocity(self):
        return self.velocity_force

    def get_passive_force(self):
        return self.passive_fiber_force * self.maximal_isometric_force

    def get_passive_force_norm(self):
        return self.passive_fiber_force

    def get_tendon_force(self):
        return self.tendon_force

    def get_active_fiber_force(self):
        self.active_fiber_force_denorm = self.active_fiber_force * self.activation * self.maximal_isometric_force
        return self.active_fiber_force_denorm

    def get_tendon_length(self):
        return self.tendon_length

    def get_maximal_fiber_velocity(self):
        return self.maximal_fiber_velocity

    def get_pennation_angle(self):
        alpha = np.arccos(self.cosalpha)
        return(alpha)

    # set function for muscle properties
    def set_tendon_stiffness(self, kT):
        self.kT = kT
        # and compute offset again
        self.get_tendon_shift()

    def set_maximal_isometric_force(self, FMo):
        self.maximal_isometric_force = FMo

    def set_optimal_fiber_length(self, lMo):
        self.optimal_fiber_length = lMo

    def set_tendon_slack_length(self, lTs):
        self.tendon_slack_length = lTs

    def set_optimal_pennation_angle(self, alpha):
        self.optimal_pennation_angle= alpha

    def set_maximal_fiber_velocity(self, vmax):
        self.maximal_fiber_velocity = vmax









