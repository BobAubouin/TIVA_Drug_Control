"""
Created on Tue Nov 22 15:46:11 2022

@author: aubouinb
"""
# Standard import
import time
import multiprocessing

# Third party imports
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from filterpy.common import Q_continuous_white_noise

# Local imports
from src.estimators import EKF
from src.controller import NMPC
import python_anesthesia_simulator as pas


def simu(Patient_info: list, style: str, MPC_param: list, EKF_param: list,
         random_PK: bool = False, random_PD: bool = False):
    """
    Simu function perform a closed-loop Propofol-Remifentanil anesthesia.

    Parameters
    ----------
    Patient_info : list
        list of patient informations, Patient_info = [Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax].
    style : str
        Either 'induction' or 'total' to describe the phase to simulate.
    MPC_param : list
        Parameters of the NMPC controller, MPC_param = [N, Nu, R, k_i].
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
        list of the signals during the simulation, data = [BIS, MAP, CO, Up, Ur, BIS_cible_MPC,
                                                           Xp_EKF, Xr_EKF, best_model_id, Xp, Xr]
    BIS_param: list
        BIS parameters of the simulated patient.

    """
    Ce50p = Patient_info[4]
    Ce50r = Patient_info[5]
    gamma = Patient_info[6]
    beta = Patient_info[7]
    E0 = Patient_info[8]
    Emax = Patient_info[9]

    ts = 2

    if not Ce50p:
        BIS_param = None
    else:
        BIS_param = [Ce50p, Ce50r, gamma, beta, E0, Emax]
    George = pas.Patient(Patient_info[:4], hill_param=BIS_param,
                         random_PK=random_PK, random_PD=random_PD, ts=ts, save_data=False)
    # Nominal parameters
    George_nominal = pas.Patient(Patient_info[:4], hill_param=None, ts=ts)
    BIS_param_nominal = George_nominal.hill_param
    BIS_param_nominal[4] = George.hill_param[4]

    Ap = George_nominal.propo_pk.continuous_sys.A
    Ar = George_nominal.remi_pk.continuous_sys.A
    Bp = George_nominal.propo_pk.continuous_sys.B
    Br = George_nominal.remi_pk.continuous_sys.B
    A_nom = block_diag(Ap, Ar)
    B_nom = block_diag(Bp, Br)

    # init state estimator
    Q = Q_continuous_white_noise(4, spectral_density=10**EKF_param[0], block_size=2)
    P0 = np.eye(8) * 10**EKF_param[1]
    estimator = EKF(A_nom, B_nom, BIS_param=BIS_param_nominal, ts=ts,
                    P0=P0, R=10**EKF_param[2], Q=Q)

    # init controller
    N_mpc = MPC_param[0]
    Nu_mpc = MPC_param[1]
    R_mpc = MPC_param[2]
    ki_mpc = MPC_param[3]
    BIS_cible = 50
    up_max = 6.67
    ur_max = 16.67
    dup_max = 0.2 * ts * 100
    dur_max = 0.4 * ts * 100

    MPC_controller = NMPC(A_nom, B_nom, BIS_param=BIS_param_nominal, ts=ts, N=N_mpc, Nu=Nu_mpc,
                          R=R_mpc, umax=[up_max, ur_max], dumax=[dup_max, dur_max],
                          dumin=[-dup_max, - dur_max], ki=0)

    if style == 'induction':
        N_simu = int(10 / ts) * 60
        BIS = np.zeros(N_simu)
        BIS_cible_MPC = np.zeros(N_simu)
        BIS_EKF = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        Xp = np.zeros((4, N_simu))
        Xr = np.zeros((4, N_simu))
        Xp_EKF = np.zeros((4, N_simu))
        Xr_EKF = np.zeros((4, N_simu))
        uP = 1e-3
        uR = 1e-3
        for i in range(N_simu):

            Dist = pas.compute_disturbances(i * ts, 'null')
            Bis, Co, Map, _ = George.one_step(uP, uR, Dist=Dist, noise=False)
            Xp[:, i] = George.propo_pk.x
            Xr[:, i] = George.remi_pk.x

            BIS[i] = Bis
            MAP[i] = Map
            CO[i] = Co
            Up[i] = uP
            Ur[i] = uR
            # estimation
            X, BIS_EKF[i] = estimator.estimate([uP, uR], BIS[i])
            Xp_EKF[:, i] = X[:4]
            Xr_EKF[:, i] = X[4:]
            # X_MPC = np.concatenate((Xp[:,i],Xr[:,i]),axis = 0)
            if i == 90:  # or (BIS_EKF[i]<50 and MPC_controller.ki==0):
                MPC_controller.ki = ki_mpc
                BIS_cible = 50
            X = np.clip(X, a_min=0, a_max=1e10)
            uP, uR = MPC_controller.one_step(X, BIS_cible, BIS_EKF[i])
            BIS_cible_MPC[i] = MPC_controller.internal_target

    elif style == 'total':
        N_simu = int(60 / ts) * 60
        BIS = np.zeros(N_simu)
        BIS_EKF = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        Xp = np.zeros((4, N_simu))
        Xr = np.zeros((4, N_simu))
        Xp_EKF = np.zeros((4, N_simu))
        Xr_EKF = np.zeros((4, N_simu))
        uP = 1
        uR = 1
        for i in range(N_simu):

            Dist = pas.compute_disturbances(i * ts, 'realistic')
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
            uP, uR = MPC_controller.one_step(X, BIS_cible, BIS_EKF[i])

    error = BIS - BIS_cible
    IAE = np.sum(np.abs(error))

    return (IAE, [BIS, MAP, CO, Up, Ur, BIS_cible_MPC, Xp_EKF, Xr_EKF],
            George.hill_param)


# %% Inter patient variability
# Simulation parameter
phase = 'induction'
Number_of_patient = 500
MPC_param = [30, 30, 10**(0.6)*np.diag([10, 1]), 0.02]
EKF_param = [1, -1, -1]


def one_simu(i):
    """Cost of one simulation, i is the patient index."""
    # Generate random patient information with uniform distribution
    np.random.seed(i)
    print(i)
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=1)

    Patient_info = [age, height, weight, gender] + [None] * 6
    iae, data, BIS_param = simu(Patient_info, 'induction', MPC_param, EKF_param,
                                random_PK=True, random_PD=True,)
    return ([iae, data, BIS_param, i])


t0 = time.time()
pool_obj = multiprocessing.Pool(8)
result = pool_obj.map(one_simu, range(0, Number_of_patient))
pool_obj.close()
pool_obj.join()

df = pd.DataFrame()

for i in range(Number_of_patient):
    print(i)

    data = result[i][1]
    name = ['BIS', 'MAP', 'CO', 'Up', 'Ur']
    dico = {str(i) + '_' + name[j]: data[j] for j in range(5)}
    df = pd.concat([df, pd.DataFrame(dico)], axis=1)

df.to_csv("./Results_data/result_NMPC_n=" + str(Number_of_patient) + '.csv')
t1 = time.time()

print(t1 - t0)
