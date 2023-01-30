"""
Created on Thu Oct 13 13:05:00 2022

@author: aubouinb
"""

# Third party imports
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from filterpy.common import Q_continuous_white_noise
import numpy as np

# Local imports
from src.estimators import EKF
from src.controller import PID
from python_anesthesia_simulator import patient, disturbances


age = 24
height = 170
weight = 55
gender = 1
Ce50p = 6.5
Ce50r = 12.2
gamma = 3
beta = 0.5
E0 = 98
Emax = 95


def simu(Patient_info: list, PID_param: list, EKF_param: list, random_PK: bool = False, random_PD: bool = False):
    """
    Perform a closed-loop Propofol-Remifentanil anesthesia simulation with a PID controller.

    Parameters
    ----------
    Patient_info : list
        list of patient informations, Patient_info = [Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax].
    style : str
        Either 'induction' or 'total' to describe the phase to simulate.
    PID_param : list
        Parameters of the NMPC controller, MPC_param = [Kp, Ti, Td, ratio].
    EKF_param : list
        Parameters of the EKF, EKF_param = log_10([Q, P0, R]).
    random_PK : bool, optional
        Add uncertaintie to the PK model. The default is False.
    random_PD : bool, optional
        Add uncertainties to the PD model. The default is False.

    Returns
    -------
    IAE : float
        Integrated Absolute Error, performance index of the function
    data : list
        list of the signals during the simulation, data = [BIS, MAP, CO, up, ur]
    BIS_param: list
        BIS parameters of the simulated patient.

    """
    age = Patient_info[0]
    height = Patient_info[1]
    weight = Patient_info[2]
    gender = Patient_info[3]
    Ce50p = Patient_info[4]
    Ce50r = Patient_info[5]
    gamma = Patient_info[6]
    beta = Patient_info[7]
    E0 = Patient_info[8]
    Emax = Patient_info[9]

    ts = 5

    BIS_param = [Ce50p, Ce50r, gamma, beta, E0, Emax]
    George = patient.Patient(age, height, weight, gender, BIS_param=BIS_param,
                             Random_PK=random_PK, Random_PD=random_PD, Ts=ts)

    # Nominal parameters
    George_nominal = patient.Patient(age, height, weight, gender, BIS_param=[None]*6, Ts=ts)
    BIS_param_nominal = George_nominal.BisPD.BIS_param
    BIS_param_nominal[4] = George.BisPD.BIS_param[4]

    Ap = George_nominal.PropoPK.A
    Ar = George_nominal.RemiPK.A
    Bp = George_nominal.PropoPK.B
    Br = George_nominal.RemiPK.B
    A_nom = block_diag(Ap, Ar)
    B_nom = block_diag(Bp, Br)

    # init state estimator
    Q = Q_continuous_white_noise(4, spectral_density=10**EKF_param[0], block_size=2)
    P0 = np.eye(8) * EKF_param[1]
    estimator = EKF(A_nom, B_nom, BIS_param=BIS_param_nominal, ts=ts,
                    P0=P0, R=EKF_param[2], Q=Q)

    # init controller
    BIS_cible = 50
    up_max = 6.67
    ur_max = 16.67
    ratio = PID_param[3]
    PID_controller = PID(Kp=PID_param[0], Ti=PID_param[1], Td=PID_param[2],
                         N=5, Ts=1, umax=max(up_max, ur_max/ratio), umin=0)

    N_simu = 30*60
    BIS = np.zeros(N_simu)
    BIS_EKF = np.zeros(N_simu)
    MAP = np.zeros(N_simu)
    CO = np.zeros(N_simu)
    Up = np.zeros(N_simu)
    Ur = np.zeros(N_simu)
    Xp = np.ones((4, N_simu))
    Xr = np.ones((4, N_simu))
    Xp_EKF = np.zeros((4, N_simu))*1
    Xr_EKF = np.zeros((4, N_simu))*1
    uP = 0
    for i in range(N_simu):
        # if i == 100:
        #     print("break")

        uR = min(ur_max, max(0, uP*ratio))  # + 0.5*max(0,np.sin(i/15)) + 0.2*int(i>200)
        uP = min(up_max, max(0, uP))  # + 0.5*max(0,np.cos(i/8)) + 0.1*int(i>400)
        Dist = disturbances.compute_disturbances(i, 'step')
        Bis, Co, Map, _, _ = George.one_step(uP, uR, Dist=Dist, noise=True)
        Xp[:, i] = George.PropoPK.x.T[0]
        Xr[:, i] = George.RemiPK.x.T[0]

        BIS[i] = Bis
        MAP[i] = Map
        CO[i] = Co
        Up[i] = uP
        Ur[i] = uR
        # estimation
        X, BIS_EKF[i] = estimator.estimate([uP, uR], BIS[i])
        Xp_EKF[:, i] = X[:4]
        Xr_EKF[:, i] = X[4:]
        uP = PID_controller.one_step(BIS_EKF[i], BIS_cible)

    IAE = np.sum(np.abs(Xp_EKF[3, :] - Xp[3, :])) + np.sum(np.abs(Xr_EKF[3, :] - Xr[3, :]))

    return IAE, [Xp_EKF, Xp, Xr_EKF, Xr, BIS, BIS_EKF]


EKF_param = [1, -1, 1]
# %% Patient table:
# index, Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax
Patient_table = [[1,  40, 163, 54, 0, 4.73, 24.97,  1.08,  0.30, 97.86, 89.62],
                 [2,  36, 163, 50, 0, 4.43, 19.33,  1.16,  0.29, 89.10, 98.86],
                 [3,  28, 164, 52, 0, 4.81, 16.89,  1.54,  0.14, 93.66, 94.],
                 [4,  50, 163, 83, 0, 3.86, 20.97,  1.37,  0.12, 94.60, 93.2],
                 [5,  28, 164, 60, 1, 5.22, 18.95,  1.21,  0.68, 97.43, 96.21],
                 [6,  43, 163, 59, 0, 3.41, 23.26,  1.34,  0.58, 85.33, 97.07],
                 [7,  37, 187, 75, 1, 4.83, 15.21,  1.84,  0.13, 91.87, 90.84],
                 [8,  38, 174, 80, 0, 4.36, 13.86,  2.23,  1.05, 97.45, 96.36],
                 [9,  41, 170, 70, 0, 2.97, 14.20,  1.89,  0.16, 85.83, 94.6],
                 [10, 37, 167, 58, 0, 6.02, 23.47,  1.27,  0.77, 95.18, 88.17],
                 [11, 42, 179, 78, 1, 3.79, 22.25,  2.35,  1.12, 98.02, 96.95],
                 [12, 34, 172, 58, 0, 5.70, 18.64,  2.02,  0.40, 99.57, 96.94],
                 [13, 38, 169, 65, 0, 4.64, 19.50,  1.43,  0.48, 93.82, 94.40]]

# %% show results on mean patient


param_PID = [0.016894,  325.329175,  7.682692, 2]
Patient_info = Patient_table[9-1][1:]

EKF_param = [1, -1, 1]

iae, data = simu(Patient_info, param_PID, EKF_param)

Xp_EKF = data[0]
Xp = data[1]
Xr_EKF = data[2]
Xr = data[3]
BIS = data[4]
BIS_EKF = data[5]

fig, axs = plt.subplots(8, figsize=(14, 16))
for i in range(4):
    axs[i].plot(Xp[i, :], '-')
    axs[i].plot(Xp_EKF[i, :], '-')
    axs[i].set(xlabel='t', ylabel='$xp_' + str(i+1) + '$')
    plt.grid()
    axs[i+4].plot(Xr[i, :], '-')
    axs[i+4].plot(Xr_EKF[i, :], '-')
    axs[i+4].set(xlabel='t', ylabel='$xr_' + str(i+1) + '$')
plt.show()

plt.figure()
plt.plot(BIS, label='Measure')
plt.plot(BIS_EKF, label='Estimation')
plt.grid()
plt.legend()
plt.show()
