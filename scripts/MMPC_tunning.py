"""
Created on Wed Dec  7 09:51:25 2022

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
import matplotlib.pyplot as plt

# Local imports
from src.estimators import EKF
from src.controller import NMPC, MMPC
import python_anesthesia_simulator as pas


def simu(Patient_info: list, style: str, MPC_param: list, EKF_param: list, MMPC_param: list,
         random_PK: bool = False, random_PD: bool = False) -> (float, list, list):
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
    EKF_param : list
        Parameters of the EKF, EKF_param = log_10([Q, P0, R]).
    MMPC_param: list
        Parameters of the model choice in the MMPC, MMPC_param = [window_length, alpha, beta, lambda, hysteresis].
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
    George = pas.Patient(Patient_info[:4], hill_param=BIS_param,
                         random_PK=random_PK, random_PD=random_PD, ts=ts, save_data=False)

    # Nominal parameters
    George_nominal = pas.Patient(Patient_info[:4], hill_param=None, ts=ts)
    BIS_param_nominal = George_nominal.hill_param
    # BIS_param_nominal[4] = George.hill_param[4]

    Ap = George_nominal.propo_pk.continuous_sys.A
    Ar = George_nominal.remi_pk.continuous_sys.A
    Bp = George_nominal.propo_pk.continuous_sys.B
    Br = George_nominal.remi_pk.continuous_sys.B
    A_nom = block_diag(Ap, Ar)
    B_nom = block_diag(Bp, Br)

    cv_c50p = 0.182
    cv_c50r = 0.888
    cv_gamma = 0.304

    # estimation of log normal standard deviation
    w_c50p = np.sqrt(np.log(1+cv_c50p**2))
    w_c50r = np.sqrt(np.log(1+cv_c50r**2))
    w_gamma = np.sqrt(np.log(1+cv_gamma**2))

    c50p_list = BIS_param_nominal[0]*np.exp([-2*w_c50p, 0, w_c50p])
    c50r_list = BIS_param_nominal[1]*np.exp([-3*w_c50r, -1*w_c50r, 0, w_c50r])
    gamma_list = BIS_param_nominal[2]*np.exp([-2*w_gamma, 0, w_gamma])
    BIS_parameters = []
    for c50p in c50p_list:
        for c50r in c50r_list:
            for gamma in gamma_list:
                BIS_parameters.append([c50p, c50r, gamma]+BIS_param_nominal[3:])

    model_number = len(BIS_parameters)

    # State estimator parameters
    Q = Q_continuous_white_noise(4, spectral_density=10**EKF_param[0], block_size=2)  # np.eye(8) * 10**EKF_param[0]  #
    P0 = np.eye(8) * 10**EKF_param[1]
    Estimator_list = []

    # Controller parameters
    N_mpc = MPC_param[0]
    Nu_mpc = MPC_param[1]
    R_mpc = MPC_param[2]
    ki_mpc = MPC_param[3]
    BIS_cible = 50
    up_max = 6.67
    ur_max = 13.67
    dup_max = 0.2 * ts * 100
    dur_max = 0.4 * ts * 100

    Controller_list = []
    for i in range(model_number):
        Estimator_list.append(EKF(A_nom, B_nom, BIS_param=BIS_parameters[i],
                                  ts=ts, P0=P0, R=10**EKF_param[2], Q=Q))
        Controller_list.append(NMPC(A_nom, B_nom,
                                    BIS_param=BIS_parameters[i],
                                    ts=ts, N=N_mpc, Nu=Nu_mpc, R=R_mpc,
                                    umax=[up_max, ur_max],
                                    dumax=[dup_max, dur_max],
                                    dumin=[-dup_max, - dur_max],
                                    ki=0))

    Controller = MMPC(Estimator_list, Controller_list, hysteresis=MMPC_param[4], window_length=MMPC_param[0],
                      best_init=13, alpha=MMPC_param[1], beta=MMPC_param[2], lambda_p=MMPC_param[3])

    if style == 'induction':
        N_simu = int(10 / ts) * 60
        BIS = np.zeros(N_simu)
        BIS_cible_MPC = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        best_model_id = np.zeros(N_simu)
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
            if i == N_simu - 1:
                break
            # control
            if i > 90/ts:
                # MMPC.controller.ki = ki_mpc
                for j in range(model_number):
                    Controller.controller_list[j].ki = ki_mpc
            U, best_model = Controller.one_step([uP, uR], Bis)
            Xp_EKF[:, i] = Estimator_list[13].x[:4]
            Xr_EKF[:, i] = Estimator_list[13].x[4:]
            best_model_id[i] = best_model
            uP = U[0]
            uR = U[1]
            Up[i] = uP
            Ur[i] = uR

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
            Bis, Co, Map, _ = George.one_step(uP, uR, Dist=Dist, noise=False)
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
    print(np.array(BIS_parameters[best_model]).round(3))
    print(np.array(George.hill_param).round(3))
    return(IAE, [BIS, MAP, CO, Up, Ur, BIS_cible_MPC, Xp_EKF, Xr_EKF, best_model_id, Xp, Xr], George.hill_param)


# %% Table simultation
Patient_table = pd.read_csv('./scripts/Patient_table.csv')
# Simulation parameters

MPC_param = [30, 30, 10**(0.6)*np.diag([10, 1]), 0.02]
EKF_param = [1, -1, -1]
MMPC_param = [30, 0, 1, 0.05, 30]
phase = 'induction'
ts = 2


def one_simu(i):
    """Cost of one simulation, i is the patient index."""
    # Generate random patient information with uniform distribution

    Patient_info = Patient_table.loc[i-1].to_numpy()[1:]

    IAE, data, BIS_param = simu(Patient_info, phase, MPC_param, EKF_param, MMPC_param)
    return [IAE, data, BIS_param, i]


# t0 = time.time()
# pool_obj = multiprocessing.Pool(8)
# result = pool_obj.map(one_simu, range(1, 15))
# pool_obj.close()
# pool_obj.join()
# t1 = time.time()
# print('one_step time: ' + str((t1-t0)*8/(len(result[0][1][0])*13)))

IAE_list = []
TT_list = []
ST10_list = []
p1 = figure(width=900, height=300)
p2 = figure(width=900, height=300)
p3 = figure(width=900, height=300)
p4 = figure(width=900, height=300)
time_simulation = []
for i in range(16):  # len(Patient_table)):
    print(i+1)
    Patient_info = Patient_table.loc[i].to_numpy()[1:]
    t0 = time.time()
    IAE, data, BIS_param = simu(Patient_info, phase, MPC_param, EKF_param, MMPC_param)
    t1 = time.time()
    time_simulation.append(t1-t0)
    # IAE = result[i][0]
    # data = result[i][1]
    # BIS_param = result[i][2]

    Xp_EKF = data[6]
    Xp = data[9]
    Xr_EKF = data[7]
    Xr = data[10]
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
    source = pd.DataFrame(data=data[0], columns=['BIS'])
    source.insert(len(source.columns), "time", np.arange(0, len(data[0]))*ts/60)
    source.insert(len(source.columns), "Ce50_P", BIS_param[0])
    source.insert(len(source.columns), "Ce50_R", BIS_param[1])
    source.insert(len(source.columns), "gamma", BIS_param[2])
    source.insert(len(source.columns), "beta", BIS_param[3])
    source.insert(len(source.columns), "E0", BIS_param[4])
    source.insert(len(source.columns), "Emax", BIS_param[5])

    plot = p1.line(x='time', y='BIS', source=source)
    tooltips = [('Ce50_P', "@Ce50_P"), ('Ce50_R', "@Ce50_R"),
                ('gamma', "@gamma"), ('beta', "@beta"),
                ('E0', "@E0"), ('Emax', "@Emax")]
    p1.add_tools(HoverTool(renderers=[plot], tooltips=tooltips))
    # p1.line(np.arange(0, len(data[0])-1)*ts/60, data[5][0:-1],
    #         legend_label='internal target', line_color="#f46d43")
    p2.line(np.arange(0, len(data[0]))*ts/60,
            data[1], legend_label='MAP (mmgh)')
    p2.line(np.arange(0, len(data[0]))*ts/60, data[2]*10,
            legend_label='CO (cL/min)', line_color="#f46d43")
    p3.line(np.arange(0, len(data[3]))*ts/60, data[3],
            line_color="#006d43", legend_label='propofol (mg/min)')
    p3.line(np.arange(0, len(data[4]))*ts/60, data[4],
            line_color="#f46d43", legend_label='remifentanil (ng/min)')
    p3.line(np.arange(0, len(data[8]))*ts/60, data[8], legend_label='Best model id')
    p4.line(data[9][3], data[10][3])
    TT, BIS_NADIR, ST10, ST20, US = pas.compute_control_metrics(data[0], Ts=ts, phase=phase)
    TT_list.append(TT)
    ST10_list.append(ST10)
    IAE_list.append(IAE)

print('Mean TT:' + str(np.mean(TT_list)) + 's')
print(np.mean(time_simulation)/len(data[0]))
p1.title.text = 'BIS'
p3.title.text = 'Infusion rates'
p1.xaxis.axis_label = 'Time (min)'
p4.xaxis.axis_label = 'Ce_propo (µg/ml)'
p4.yaxis.axis_label = 'Ce_remi (ng/ml)'
p2.xaxis.axis_label = 'Time (min)'
p3.xaxis.axis_label = 'Time (min)'
grid = row(column(p3, p1, p2), p4)

show(grid)
