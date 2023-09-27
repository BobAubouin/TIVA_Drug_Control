"""
Created on Fri Dec  9 14:22:34 2022

@author: aubouinb
"""
# Standard import
import matplotlib
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
from progress.bar import ChargingBar
from tqdm import tqdm
from functools import partial
from itertools import product

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


# patient_id for tunning
np.random.seed(0)
case_list = np.random.randint(0, 500, 16)

# Simulation parameter
phase = 'induction'
ts = 2

MPC_param = [30, 30, 10**(1)*np.diag([10, 1])]
N_mpc_list = [20, 30, 40]
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


def one_simu(i, MPC_param, MHE_param):
    """Cost of one simulation, i is the patient index."""
    # Generate random patient information with uniform distribution
    np.random.seed(i)
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=1)

    Patient_info = [age, height, weight, gender] + [None] * 6
    IAE, _, _ = simu(Patient_info, phase, MPC_param, MHE_param,
                     random_PK=True, random_PD=True)

    return IAE


def cost(MPC_param, MHE_param):
    IAE_list = []
    for i in case_list:
        IAE_list.append(one_simu(i, MPC_param, MHE_param))
    return max(IAE_list)

# %% tunning


MPC_param_list = list(product(N_mpc_list, R_list))

with mp.Pool(8) as pool:
    IAE_list = list(tqdm(pool.map(partial(cost, MHE_param=MHE_param), MPC_param_list), total=len(MPC_param_list)))

N_value = [MPC_param_list[i][0] for i in range(len(MPC_param_list))]
R_value = [np.log10(MPC_param_list[i][1]) for i in range(len(MPC_param_list))]

df = pd.DataFrame({'N': N_value, 'R': R_value, 'IAE': IAE_list})
df.to_csv('Results_data/tunning_MPC.csv')

# %% plot tunning

# plot config
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font = {'family': 'serif',
        'weight': 'normal',
        'size': 16}

plt.rc('text', usetex=True)
matplotlib.rc('font', **font)


# %% Plot results

fig, host = plt.subplots(figsize=(10, 5))

ynames = ['$N_{MPC}$', '$\log_{10}(R)$', 'IAE']


# organize the data
ys = np.array([N_value, R_value, IAE_list]).T
ymins = ys.min(axis=0)
ymaxs = ys.max(axis=0)
dys = ymaxs - ymins
ymins -= dys * 0.05  # add 5% padding below and above
ymaxs += dys * 0.05
dys = ymaxs - ymins

# transform all data to be compatible with the main axis
zs = np.zeros_like(ys)
zs[:, 0] = ys[:, 0]
zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]


axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
for i, ax in enumerate(axes):
    ax.set_ylim(ymins[i], ymaxs[i])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax != host:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    # if i == 0:
    #     ax.set_yscale('log')

host.set_xlim(0, ys.shape[1] - 1)
host.set_xticks(range(ys.shape[1]))
host.set_xticklabels(ynames, fontsize=14)
host.tick_params(axis='x', which='major', pad=7)
host.spines['right'].set_visible(False)
host.xaxis.tick_top()
# host.set_title('Parallel Coordinates Plot', fontsize=18)

up = np.array([248, 123, 0])/255
down = np.array([0, 200, 14])/255
min = min(IAE_list)
max = max(IAE_list)
for j in range(len(ys)):
    # to just draw straight lines between the axes:
    # host.plot(range(ys.shape[1]), zs[j,:], c=colors[(category[j] - 1) % len(colors) ])

    # create bezier curves
    # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
    #   at one third towards the next axis; the first and last axis have one less control vertex
    # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
    # y-coordinate: repeat every point three times, except the first and last only twice
    verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                     np.repeat(zs[j, :], 3)[1:-1]))
    # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    alpha = (IAE_list[j] - min)/(max-min)
    if alpha == 0:
        patch = patches.PathPatch(path, facecolor='none', lw=2, edgecolor='#5d88f9ff')
        patch_min = patch
    else:
        patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=tuple(up * alpha + down*(1-alpha)), alpha=0.5)
        host.add_patch(patch)
host.add_patch(patch_min)
for i, ax in enumerate(axes):
    if i < len(ys[0])-1:
        ax.yaxis.set_ticks(np.round(np.unique(ys[:, i]), 1))
        ax.yaxis.set_ticklabels(np.round(np.unique(ys[:, i]), 1))

plt.tight_layout()
plt.savefig(f'Results_Images/tunning_MPC.pdf', bbox_inches='tight', format='pdf')
plt.show()
