"""
Created on Fri Dec  9 14:22:34 2022

@author: aubouinb
"""
# Standard import
import matplotlib
import pathlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import time
import multiprocessing as mp

# Third party imports
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from filterpy.common import Q_continuous_white_noise
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import HoverTool
from tqdm import tqdm
from functools import partial
from itertools import product

# Local imports
from pyswarm import pso
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
        Parameters of the NMPC controller, MPC_param = [N, R].
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

    BIS_param_nominal = George.hill_param

    Ap = George_nominal.propo_pk.continuous_sys.A[:4, :4]
    Ar = George_nominal.remi_pk.continuous_sys.A[:4, :4]
    Bp = George_nominal.propo_pk.continuous_sys.B[:4]
    Br = George_nominal.remi_pk.continuous_sys.B[:4]
    A_nom = block_diag(Ap, Ar)
    B_nom = block_diag(Bp, Br)

    # Controller parameters
    N_mpc = MPC_param[0]
    R_mpc = MPC_param[1]
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
    Controller = NMPC(A_nom, B_nom, BIS_param_nominal, ts=ts, N=N_mpc, Nu=N_mpc, R=R_mpc, umax=[up_max, ur_max],
                      dumax=[dup_max, dur_max], dumin=[-dup_max, - dur_max], bool_u_eq=True)

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


def one_simu(i, MPC_param, MHE_param, phase):
    """Cost of one simulation, i is the patient index."""
    # Generate random patient information with uniform distribution
    np.random.seed(i)
    print(i)
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=1)

    Patient_info = [age, height, weight, gender] + [None] * 6
    IAE, _, _ = simu(Patient_info, phase, MPC_param, MHE_param,
                     random_PK=True, random_PD=True)

    return IAE


def cost(R, N_mpc, MHE_param, case_list, phase):
    with mp.Pool(mp.cpu_count()) as p:
        partial_f = partial(one_simu, MPC_param=[N_mpc, 10**R*np.diag([10, 1])], MHE_param=MHE_param, phase=phase)
        IAE_list = list(p.map(partial_f, case_list))

    return max(IAE_list)


if __name__ == '__main__':
    # patient_id for tunning
    np.random.seed(0)
    case_list = np.random.randint(0, 500, 16)

    # Simulation parameter
    phase = 'induction'
    ts = 2

    MPC_param = [30, 30, 10**(1)*np.diag([10, 1])]
    N_mpc = 30
    R_list = [el*np.diag([10, 1]) for el in np.logspace(0, 2, 3)]

    gamma = 1.e-2
    theta = [gamma, 800, 100, 0.005]*4
    theta[4] = gamma/100
    # theta[12] = 1e-5
    # theta[13] = 300
    # theta[15] = 0.1
    Q = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
    R = 1e0
    N_mhe = 25
    MHE_param = [R, Q, theta, N_mhe]

    # %% tunning

    is_reject = phase == 'maintenance'
    file = pathlib.Path(f"./scripts/optimal_parameters_MPC{'_reject' if is_reject else ''}.csv")
    if file.exists():
        param_opti = pd.read_csv(file)
    else:
        nb_point = 4
        lb, ub = 1, 3
        f_cost = partial(cost, N_mpc=N_mpc, MHE_param=MHE_param, case_list=case_list, phase=phase)
        IAE_list = list(map(f_cost, np.linspace(lb, ub, nb_point)))
        R_list = np.linspace(lb, ub, nb_point)
        id_min = np.argmin(IAE_list)
        R = R_list[id_min]

        param_opti = pd.DataFrame({'N_mpc': N_mpc, 'R': R}, index=[0])

        plt.plot(R_list, IAE_list)
        plt.savefig('./Results_Images/tunning_MPC.pdf')

        param_opti.to_csv(file)

    # %% plot tunning

    # Number_of_patient = 500
    # # Controller parameters
    # R = param_opti['R'][0]
    # N_mpc = param_opti['N_mpc'][0]
    # MPC_param = [N_mpc, 10**R*np.diag([10, 1])]

    # df = pd.DataFrame()
    # pd_param = pd.DataFrame()
    # name = ['BIS', 'MAP', 'CO', 'Up', 'Ur']

    # def one_simu(i):
    #     np.random.seed(i)
    #     # Generate random patient information with uniform distribution
    #     age = np.random.randint(low=18, high=70)
    #     height = np.random.randint(low=150, high=190)
    #     weight = np.random.randint(low=50, high=100)
    #     gender = np.random.randint(low=0, high=2)

    #     Patient_info = [age, height, weight, gender] + [None] * 6
    #     _, data, bis_param = simu(Patient_info, phase, MPC_param, MHE_param, random_PD=True, random_PK=True)
    #     return Patient_info, data, bis_param

    # with mp.Pool(mp.cpu_count()) as p:
    #     r = list(tqdm(p.imap(one_simu, range(Number_of_patient)), total=Number_of_patient))

    # for i in tqdm(range(Number_of_patient)):
    #     Patient_info, data, bis_param = r[i]
    #     age = Patient_info[0]
    #     height = Patient_info[1]
    #     weight = Patient_info[2]
    #     gender = Patient_info[3]

    #     dico = {str(i) + '_' + name[j]: data[j] for j in range(5)}
    #     df = pd.concat([df, pd.DataFrame(dico)], axis=1)

    #     dico = {'age': [age],
    #             'height': [height],
    #             'weight': [weight],
    #             'gender': [gender],
    #             'C50p': [bis_param[0]],
    #             'C50r': [bis_param[1]],
    #             'gamma': [bis_param[2]],
    #             'beta': [bis_param[3]],
    #             'Emax': [bis_param[4]],
    #             'E0': [bis_param[5]]}
    #     pd_param = pd.concat([pd_param, pd.DataFrame(dico)], axis=0)

    # if phase == 'maintenance':
    #     df.to_csv("./Results_data/result_MPC_maintenance_n=" + str(Number_of_patient) + '.csv')
    # elif phase == 'induction':
    #     df.to_csv("./Results_data/result_MPC_induction_n=" + str(Number_of_patient) + '.csv')
    # else:
    #     df.to_csv("./Results_data/result_MPC_total_n=" + str(Number_of_patient) + '.csv')
