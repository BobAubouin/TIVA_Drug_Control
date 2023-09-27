"""
Created on Fri Dec  9 14:22:34 2022

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
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import HoverTool
from tqdm import tqdm

# Local imports
from estimators import MHE_integrator as MHE
from controller import NMPC_integrator as NMPC
import python_anesthesia_simulator as pas


def simu(Patient_info: list, style: str, MPC_param: list, MHE_param: list,
         random_PK: bool = False, random_PD: bool = False) -> tuple[float, list, list]:
    """
    Simu function perform a closed-loop Propofol-Remifentanil to BIS anesthesia.

    Parameters
    ----------
    Patient_info : list
        list of patient informations, Patient_info = [Age[yr], H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax].
    style : str
        Either 'induction' or 'total' to describe the phase to simulate.
    MPC_param : list
        Parameters of the NMPC controller, MPC_param = [N, Nu, R, k_i].
    MHE_param : list
        Parameters of the MHE, [R, Q, theta, Nmhe]
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
        # BIS_param = [3.5, 19.3, 1.43, 0, 97.4, 97.4]
    George = pas.Patient(Patient_info[:4], hill_param=BIS_param, random_PK=random_PK,
                         random_PD=random_PD, ts=ts, save_data_bool=False)

    # Nominal parameters
    George_nominal = pas.Patient(Patient_info[:4], hill_param=None, ts=ts)
    BIS_param_nominal = George_nominal.hill_param
    # BIS_param_nominal[4] = George.hill_param[4]

    Ap = George_nominal.propo_pk.continuous_sys.A[:4, :4]
    Ar = George_nominal.remi_pk.continuous_sys.A[:4, :4]
    Bp = George_nominal.propo_pk.continuous_sys.B[:4]
    Br = George_nominal.remi_pk.continuous_sys.B[:4]
    A_nom = block_diag(Ap, Ar)
    B_nom = block_diag(Bp, Br)

    # Controller parameters
    N_mpc = MPC_param[0]
    Nu_mpc = MPC_param[1]
    R_mpc = MPC_param[2]
    BIS_cible = 50
    up_max = 6.67
    ur_max = 13.67
    dup_max = 0.2 * ts * 100
    dur_max = 0.4 * ts * 100

    # MHE parameters
    R_mhe = MHE_param[0]
    Q_mhe = MHE_param[1]
    theta_mhe = MHE_param[2]
    N_mhe = MHE_param[3]

    # instance of the controller and the estimator
    Controller = NMPC(A_nom, B_nom, BIS_param_nominal, ts=ts, N=N_mpc, Nu=Nu_mpc, R=R_mpc, umax=[up_max, ur_max],
                      dumax=[dup_max, dur_max], dumin=[-dup_max, - dur_max])

    Estimator = MHE(A_nom, B_nom, BIS_param_nominal, ts=ts, R=R_mhe, Q=Q_mhe, theta=theta_mhe, N_MHE=N_mhe)

    if style == 'induction':
        N_simu = int(10 / ts) * 60
        BIS = np.zeros(N_simu)
        BIS_cible_MPC = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        Xp = np.zeros((4, N_simu))
        Xr = np.zeros((4, N_simu))
        x_mhe = np.zeros((12, N_simu))
        bis_mhe = np.zeros(N_simu)
        uP = 1e-3
        uR = 1e-3
        step_time_max = 0
        for i in range(N_simu):
            Dist = pas.compute_disturbances(i * ts, 'null')
            Bis, Co, Map, _ = George.one_step(uP, uR, dist=Dist, noise=False)
            Xp[:, i] = George.propo_pk.x[:4]
            Xr[:, i] = George.remi_pk.x[:4]
            BIS[i] = Bis[0]
            MAP[i] = Map[0]
            CO[i] = Co[0]
            if i == N_simu - 1:
                break
            # control
            start = time.perf_counter()
            x_mhe[:, [i]], bis_mhe[i] = Estimator.one_step(Bis, [uP, uR])
            x_mpc = np.hstack((x_mhe[:8, i], x_mhe[-1, i]))
            uP, uR = Controller.one_step(x_mpc, BIS_cible, bis_param=list(x_mhe[8:11, i]))
            end = time.perf_counter()
            Up[i] = uP
            Ur[i] = uR
            step_time_max = max(step_time_max, end - start)
    elif style == 'total':
        N_simu = int(30 / ts) * 60
        BIS_cible_MPC = np.zeros(N_simu)
        best_model_id = np.zeros(N_simu)
        Xp_EKF = np.zeros((4 * model_number, N_simu))
        Xr_EKF = np.zeros((4 * model_number, N_simu))
        uP = 1e-3
        uR = 1e-3
        for i in range(N_simu):

            Dist = pas.compute_disturbances(i*ts, 'step')
            Bis, Co, Map, _ = George.one_step(uP, uR, dist=Dist, noise=False)
            if i == N_simu - 1:
                break
            # control
            if i > 120/ts:
                for j in range(model_number):
                    MMPC.controller_list[j].ki = ki_mpc
            U, best_model = MMPC.one_step([uP, uR], Bis)
            best_model_id[i] = best_model
            uP = U[0]
            uR = U[1]

    IAE = np.sum(np.abs(BIS - BIS_cible))
    return (IAE, [BIS, MAP, CO, Up, Ur, BIS_cible_MPC, x_mhe, Xp, Xr, step_time_max], George.hill_param)

# %% Inter patient variability


Number_of_patient = 50

# Simulation parameter
phase = 'induction'
ts = 2
MPC_param = [30, 30, 10**(0.7)*np.diag([10, 1]), 0.02]

theta = [1e-3, 800, 100, 0.005]*4
theta[4] = 1e-5
theta[12] = 1e-5
theta[13] = 300
theta[15] = 0.1
Q = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
R = 1e-3
N_mhe = 30
MHE_param = [R, Q, theta, N_mhe]


def one_simu(i):
    """Cost of one simulation, i is the patient index."""
    # Generate random patient information with uniform distribution
    np.random.seed(i)
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=1)

    Patient_info = [age, height, weight, gender] + [None] * 6
    iae, data, BIS_param = simu(Patient_info, phase, MPC_param, MHE_param,
                                random_PK=True, random_PD=True)
    return [iae, data, BIS_param, i]


t0 = time.time()

pool_obj = multiprocessing.Pool(7)
result = []
for x in tqdm(pool_obj.map(one_simu, range(Number_of_patient)), total=Number_of_patient):
    result.append(x)
pool_obj.close()
pool_obj.join()


with ChargingBar(
    "Saving in progress\t", suffix="%(percent).1f%% - ETA %(eta)ds", max=Number_of_patient, color="green"
) as bar:
    df = pd.DataFrame()
    for i in range(Number_of_patient):
        data = result[i][1]
        name = ['BIS', 'MAP', 'CO', 'Up', 'Ur']
        dico = {str(i) + '_' + name[j]: data[j] for j in range(5)}
        dico['step_time_max'] = data[-1]
        df = pd.concat([df, pd.DataFrame(dico)], axis=1)
        bar.next()
    df.to_csv("./Results_data/result_MHE_NMPC_n=" + str(Number_of_patient) + '.csv')

t1 = time.time()

print(f"Time elapsed: {t1 - t0}s")
# %%
