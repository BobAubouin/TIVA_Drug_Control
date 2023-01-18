
"""
CreaTsd on Fri Dec  9 14:22:34 2022

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
import matplotlib.pyplot as plt

# Local imports
from TIVA_Drug_Control.src.estimators import EKF
from TIVA_Drug_Control.src.controller import NMPC, MPC_lin
from src import patient, disturbances


def simu(Patient_info: list, style: str, MPC_param: list, EKF_param: list,
         random_PK: bool = False, random_PD: bool = False):
    """Simu function perform a closed-loop Propofol-Remifentanil anesthesia.

        simulation with a PID controller,

    Inputs: - Patient_info: list of patient informations,
                            Patient_info = [Age, H[cm], W[kg], Gender, Ce50p,
                                            Ce50r, γ, β, E0, Emax]
            - style: either 'induction' or 'maintenance' to describe
                    the phase to simulate
            - MPC_param: parameter of the PID controller P = [N, Q, R]
            - random: bool to add uncertainty to simulate intra-patient
                        variability in the patient model

    Outputs:- IAE: Integrated Absolute Error, performance index of the function
            - data: list of the signals during the simulation
                    data = [BIS, MAP, CO, up, ur].
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

    ts = 2

    BIS_param = [Ce50p, Ce50r, gamma, beta, E0, Emax]
    George = patient.Patient(age, height, weight, gender, BIS_param=BIS_param,
                             Random_PK=random_PK, Random_PD=random_PD, Ts=ts)
    # , model_propo = 'Eleveld', model_remi = 'Eleveld')

    # Nominal parameters
    George_nominal = patient.Patient(
        age, height, weight, gender, BIS_param=[None] * 6, Ts=ts)
    BIS_param_nominal = George_nominal.BisPD.BIS_param
    BIS_param_nominal[4] = George.BisPD.BIS_param[4]
    # BIS_param_nominal[5] = George.BisPD.BIS_param[5]

    Ap = George_nominal.PropoPK.A
    Ar = George_nominal.RemiPK.A
    Bp = George_nominal.PropoPK.B
    Br = George_nominal.RemiPK.B
    A_nom = block_diag(Ap, Ar)
    B_nom = block_diag(Bp, Br)

    model_number = 3 * 3 * 3 * 2
    std_Cep = 1.34 * 1.5
    std_Cer = 5.79 * 1.5
    std_gamma = 0.73 * 1.5
    std_beta = 0.5 * 1.5

    BIS_parameters = []
    BIS_param_grid = BIS_param_nominal.copy()
    BIS_param_grid[3] = BIS_param_nominal[3] - std_beta
    for m in range(2):
        BIS_param_grid[3] += std_beta
        BIS_param_grid[0] = BIS_param_nominal[0] - 2 * std_Cep
        for i in range(3):
            BIS_param_grid[0] += std_Cep
            BIS_param_grid[1] = BIS_param_nominal[1] - 2 * std_Cer
            for j in range(3):
                BIS_param_grid[1] += std_Cer
                BIS_param_grid[2] = BIS_param_nominal[2] + 2 * std_gamma
                for k in range(3):
                    BIS_param_grid[2] -= std_gamma
                    BIS_param_grid[2] = max(1, BIS_param_grid[2])
                    BIS_parameters.append(BIS_param_grid.copy())

    # State estimator parameters
    Q = Q_continuous_white_noise(
        4, spectral_density=10**EKF_param[0], block_size=2)
    P0 = np.diag([1] * 8) * 10**EKF_param[1]
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
                                    dymin=0, ki=0))

    if style == 'induction':
        N_simu = int(10 / ts) * 60
        BIS = np.zeros(N_simu)
        BIS_cible_MPC = np.zeros(N_simu)
        BIS_EKF = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        best_model_id = np.zeros(N_simu)
        Xp = np.zeros((4, N_simu))
        Xr = np.zeros((4, N_simu))
        Xp_EKF = np.zeros((4 * model_number, N_simu))
        Xr_EKF = np.zeros((4 * model_number, N_simu))
        uP = 1e-3
        uR = 1e-3
        error = np.zeros(model_number)
        idx_best = 13
        n_pred = 5
        epsilon_hysteresis = 1
        for i in range(N_simu):

            Dist = disturbances.compute_disturbances(i * ts, 'null')
            Bis, Co, Map = George.one_step(uP, uR, Dist=Dist, noise=False)
            Xp[:, i] = George.PropoPK.x.T[0]
            Xr[:, i] = George.RemiPK.x.T[0]

            BIS[i] = min(100, Bis)
            MAP[i] = Map[0, 0]
            CO[i] = Co[0, 0]
            if i == N_simu - 1:
                break
            # estimation
            X_estimate_temp = np.zeros((8, model_number))
            BIS_EKF_temp = np.zeros(model_number)
            Up_temp = np.zeros(model_number)
            Ur_temp = np.zeros(model_number)
            for j in range(model_number):
                (X_estimate_temp[:, j],
                 BIS_EKF_temp[j]) = Estimator_list[j].estimate([uP, uR], BIS[i])

                Xp_EKF[j * 4:(j + 1) * 4, i] = X_estimate_temp[:4, j]
                Xr_EKF[j * 4:(j + 1) * 4, i] = X_estimate_temp[4:, j]

                if i >= n_pred:

                    x = np.concatenate(((Xp_EKF[j * 4:(j + 1) * 4, i - n_pred],
                                         Xr_EKF[j * 4:(j + 1) * 4, i - n_pred])), axis=0)
                    bis_pred = Estimator_list[j].predict_from_state(x=x,
                                                                    up=Up[i - n_pred:i],
                                                                    ur=Ur[i - n_pred:i])
                    error[j] = np.sum(np.abs(bis_pred - BIS[i - n_pred:i + 1]))
                else:
                    error[j] = 0

                # X_MPC = np.concatenate((Xp[:,i],Xr[:,i]),axis = 0)
                if i == 20:  # or (BIS_EKF[i]<50 and MPC_controller.ki==0):
                    Controller_list[j].ki = ki_mpc

                X_estimate_temp[:, j] = np.clip(
                    X_estimate_temp[:, j], a_min=0, a_max=1e10)

            BIS_cible_MPC[i] = Controller_list[0].internal_target
            error_min = min(error)
            idx_best_list = [i for i, j in enumerate(error) if j == error_min]
            idx_best_new = idx_best_list[0]
            if abs(error[idx_best_new] - error[idx_best]) > epsilon_hysteresis:
                idx_best = idx_best_new

            Controller_list[idx_best].U_prec[0:2] = [uP, uR]
            uP, uR, b = Controller_list[idx_best].one_step(
                X_estimate_temp[:, idx_best], BIS_cible, BIS_EKF_temp[idx_best])
            # for j in range(model_number):
            #     if j != idx_best:
            #         if Controller_list[j].internal_target is None:
            #             Controller_list[j].internal_target = BIS_cible
            #         Controller_list[j].internal_target += Controller_list[j].ki * \
            #             (BIS_cible - BIS_EKF_temp[j])

            Up[i] = uP
            Ur[i] = uR
            best_model_id[i] = idx_best

            if i >= n_pred:
                n_pred += 1
                n_pred = min(n_pred, 30)
                epsilon_hysteresis = n_pred * 3

    # elif style == 'total':
    #     N_simu = int(20/ts)*60
    #     BIS = np.zeros(N_simu)
    #     BIS_cible_MPC = np.zeros(N_simu)
    #     BIS_EKF = np.zeros(N_simu)
    #     MAP = np.zeros(N_simu)
    #     CO = np.zeros(N_simu)
    #     Up = np.zeros(N_simu)
    #     Ur = np.zeros(N_simu)
    #     Xp = np.zeros((4, N_simu))
    #     Xr = np.zeros((4, N_simu))
    #     Xp_EKF = np.zeros((4, N_simu))
    #     Xr_EKF = np.zeros((4, N_simu))
    #     L = np.zeros(N_simu)
    #     uP = 1
    #     uR = 1
    #     for i in range(N_simu):
    #         # if i == 100:
    #         #     print("break")

    #         Dist = disturbances.compute_disturbances(i*ts, 'step')
    #         Bis, Co, Map = George.one_step(uP, uR, Dist=Dist, noise=False)
    #         Xp[:, i] = George.PropoPK.x.T[0]
    #         Xr[:, i] = George.RemiPK.x.T[0]

    #         BIS[i] = min(100, Bis)
    #         MAP[i] = Map[0, 0]
    #         CO[i] = Co[0, 0]
    #         Up[i] = uP
    #         Ur[i] = uR
    #         # estimation
    #         X, BIS_EKF[i] = estimator.estimate([uP, uR], BIS[i])
    #         Xp_EKF[:, i] = X[:4]
    #         Xr_EKF[:, i] = X[4:]
    #         uP, uR = MPC_controller.one_step(X, BIS_cible, BIS_EKF[i])
    #         BIS_cible_MPC[i] = MPC_controller.internal_target

    IAE = np.sum(np.abs(BIS - BIS_cible))
    return(IAE, [BIS, MAP, CO, Up, Ur, BIS_cible_MPC, Xp_EKF, Xr_EKF, best_model_id],
           George.BisPD.BIS_param)

# %% Inter patient variability


# Simulation parameter
phase = 'induction'
Number_of_patient = 128
# MPC_param = [30, 30, 1, 10*np.diag([2,1]), 0.1]

# param_opti = pd.read_csv('optimal_parameters_MPC_lin.csv')
# EKF_param = [0, 1, float(param_opti['R_EKF'])]
# param_opti = [int(param_opti['N']), int(param_opti['Nu']),
# float(param_opti['R']), float(param_opti['ki'])]
# MPC_param = [param_opti[0], param_opti[1], 1,
# 10**param_opti[2]*np.diag([10,1]), param_opti[3]]

MPC_param = [30, 30, 2, 1e-2]
EKF_param = [0, 0, 1]


MPC_param = [20, 20, 10**(1.5) * np.diag([3, 1]), 1e-2]
EKF_param = [1, 1, 1]
phase = 'induction'


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
    iae, data, BIS_param = simu(Patient_info, phase, MPC_param, EKF_param,
                                random_PK=True, random_PD=True)
    return [iae, data, BIS_param, i]


t0 = time.time()
result = one_simu(108)
t1 = time.time()
print(t1 - t0)

p1 = figure(width=900, height=300)
p2 = figure(width=900, height=300)
p3 = figure(width=900, height=300)
data = result[1]
Time = np.arange(0, len(data[1]))*2/60
p1.line(Time, data[0], legend_label='BIS')
# p1.line(np.arange(0,len(data[0]))*5/60, data[5], legend_label='internal target', line_color="#f46d43")
p2.line(Time, data[1], legend_label='MAP (mmgh)')
p2.line(Time, data[2] * 10,
        legend_label='CO (cL/min)', line_color="#f46d43")
p3.line(Time, data[3], line_color="#006d43",
        legend_label='propofol (mg/min)')
p3.line(Time, data[4], line_color="#f46d43",
        legend_label='remifentanil (ng/min)')
p3.line(Time, data[-1], legend_label='best model')
p1.title.text = 'BIS'
p3.title.text = 'Infusion raTss'
p3.xaxis.axis_label = 'Time (min)'
grid = row(column(p3, p1, p2))

show(grid)
