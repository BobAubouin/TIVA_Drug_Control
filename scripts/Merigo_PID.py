"""MISO PID control of anesthesia.

from L. Merigo, F. Padula, N. Latronico, M. Paltenghi, and A. Visioli, 
“Optimized PID control of propofol and remifentanil coadministration for general anesthesia,”
Communications in Nonlinear Science and Numerical Simulation
, vol. 72, pp. 194–212, Jun. 2019, doi: 10.1016/j.cnsns.2018.12.015.
"""

# %% import packages

# Standard import
import multiprocessing as mp
from functools import partial

# Third party imports
import numpy as np
import pandas as pd
from bokeh.models import HoverTool
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
from pyswarm import pso
from tqdm import tqdm

# Local imports
from controller import PID
import python_anesthesia_simulator as pas

# %% Define the simulation function


def simu(Patient_info: list, style: str, PID_param: list,
         random_PK: bool = False, random_PD: bool = False) -> tuple[float, list, list]:
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
    Ce50p = Patient_info[4]
    Ce50r = Patient_info[5]
    gamma = Patient_info[6]
    beta = Patient_info[7]
    E0 = Patient_info[8]
    Emax = Patient_info[9]

    if not Ce50p:
        BIS_param = None
    else:
        BIS_param = [Ce50p, Ce50r, gamma, beta, E0, Emax]
    George = pas.Patient(Patient_info[:4], hill_param=BIS_param, random_PK=random_PK,
                         random_PD=random_PD, save_data_bool=False)

    bis_params = George.hill_param
    Ce50p = bis_params[0]
    Ce50r = bis_params[1]
    gamma = bis_params[2]
    beta = bis_params[3]
    E0 = bis_params[4]
    Emax = bis_params[5]

    ts = 1
    BIS_target = 50
    up_max = 6.67
    ur_max = 16.67
    ratio = PID_param[3]
    PID_controller = PID(Kp=PID_param[0], Ti=PID_param[1], Td=PID_param[2],
                         N=5, Ts=1, umax=max(up_max, ur_max / ratio), umin=0)

    if style == 'induction':
        N_simu = 10 * 60
        BIS = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        uP = 0
        uR = min(ur_max, max(0, uP * ratio))
        for i in range(N_simu):
            Bis, Co, Map, _ = George.one_step(uP, uR, noise=False)
            BIS[i] = Bis[0]
            MAP[i] = Map[0]
            CO[i] = Co[0]
            if i == N_simu - 1:
                break
            uP = PID_controller.one_step(Bis[0], BIS_target)
            uR = min(ur_max, max(0, uP * ratio))
            uP = min(up_max, max(0, uP))
            Up[i] = uP
            Ur[i] = uR

    elif style == 'total':
        N_simu = int(25 / ts) * 60
        BIS = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        uP = 0
        uR = min(ur_max, max(0, uP * ratio))
        for i in range(N_simu):
            Dist = pas.disturbances.compute_disturbances(i * ts, 'step', end_step=15*60)
            Bis, Co, Map, _ = George.one_step(uP, uR, dist=Dist, noise=True)
            if type(Bis) == np.ndarray:
                Bis = Bis[0]
            BIS[i] = Bis
            MAP[i] = Map[0]
            CO[i] = Co[0]
            if i == N_simu - 1:
                break
            uP = PID_controller.one_step(Bis, BIS_target)
            uR = min(ur_max, max(0, uP * ratio))
            uP = min(up_max, max(0, uP))
            Up[i] = uP
            Ur[i] = uR

    elif style == 'maintenance':
        N_simu = 25 * 60  # 25 minutes
        BIS = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)

        # find equilibrium input
        uP, uR = George.find_bis_equilibrium_with_ratio(BIS_target, ratio)
        # initialize the simulator with the equilibrium input
        George.initialized_at_given_input(u_propo=uP, u_remi=uR)
        Bis = George.bis
        # initialize the PID at the equilibriium point
        PID_controller.integral_part = uP / PID_controller.Kp
        PID_controller.last_BIS = BIS_target
        for i in range(N_simu):
            uR = min(ur_max, max(0, uP * ratio))
            uP = min(up_max, max(0, uP))
            Bis, Co, Map, _ = George.one_step(uP, uR, noise=False)
            dist_bis, dist_map, dist_co = pas.disturbances.compute_disturbances(i, 'step')
            BIS[i] = min(100, Bis) + dist_bis
            MAP[i] = Map[0, 0] + dist_map
            CO[i] = Co[0, 0] + dist_co
            Up[i] = uP
            Ur[i] = uR
            uP = PID_controller.one_step(BIS[i], BIS_target)

    IAE = np.sum(np.abs(BIS - BIS_target))
    return IAE, [BIS, MAP, CO, Up, Ur], George.hill_param


# %% PSO
# Patient table:
# index, Age, H[cm], W[kg], Gender, Ce50p, Ce50r, γ, β, E0, Emax
np.random.seed(0)
case_list = np.random.randint(0, 500, 10)


# phase = 'maintenance'
phase = 'induction'


def one_simu(x, ratio, i):
    """Cost of one simulation, i is the patient index."""
    np.random.seed(i)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    Patient_info = [age, height, weight, gender] + [None] * 6
    iae, _, _ = simu(Patient_info, phase, [x[0], x[1], x[2], ratio], random_PD=True, random_PK=True)
    return iae


def cost(x, ratio):
    """Cost of the optimization, x is the vector of the PID controller.

    x = [Kp, Ti, Td]
    IAE is the maximum integrated absolut error over the patient population.
    """

    func = partial(one_simu, x, ratio)
    IAE = map(func, case_list)
    
    return max(IAE)


try:
    if phase == 'maintenance':
        param_opti = pd.read_csv('./scripts/optimal_parameters_PID_reject.csv')
    else:
        param_opti = pd.read_csv('./scripts/optimal_parameters_PID.csv')
except:
    param_opti = pd.DataFrame(columns=['ratio', 'Kp', 'Ti', 'Td'])
    for ratio in range(2, 3):
        def local_cost(x): return cost(x, ratio)
        lb = [1e-6, 100, 0.1]
        ub = [1, 300, 20]
        # Default: 100 particles as in the article
        xopt, fopt = pso(local_cost, lb, ub, debug=True, minfunc=1e-2, processes = mp.cpu_count())
        param_opti = pd.concat((param_opti, pd.DataFrame(
            {'ratio': ratio, 'Kp': xopt[0], 'Ti': xopt[1], 'Td': xopt[2]}, index=[0])), ignore_index=True)
        print(ratio)
    if phase == 'maintenance':
        param_opti.to_csv('./scripts/optimal_parameters_PID_reject.csv')
    else:
        param_opti.to_csv('./scripts/optimal_parameters_PID.csv')

# %%test on all the patients
ts = 1

ratio = 2
Number_of_patient = 500
# Controller parameters
Kp = float(param_opti.loc[param_opti['ratio'] == ratio, 'Kp'])
Ti = float(param_opti.loc[param_opti['ratio'] == ratio, 'Ti'])
Td = float(param_opti.loc[param_opti['ratio'] == ratio, 'Td'])
PID_param = [Kp, Ti, Td, ratio]


df = pd.DataFrame()
pd_param = pd.DataFrame()
name = ['BIS', 'MAP', 'CO', 'Up', 'Ur']


def one_simu(i):
    np.random.seed(i)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    Patient_info = [age, height, weight, gender] + [None] * 6
    _, data, bis_param = simu(Patient_info, phase, PID_param, random_PD=True, random_PK=True)
    return Patient_info, data, bis_param


with mp.Pool(mp.cpu_count()-2) as p:
    r = list(tqdm(p.imap(one_simu, range(Number_of_patient)), total=Number_of_patient))


for i in tqdm(range(Number_of_patient)):
    Patient_info, data, bis_param = r[i]
    age = Patient_info[0]
    height = Patient_info[1]
    weight = Patient_info[2]
    gender = Patient_info[3]

    dico = {str(i) + '_' + name[j]: data[j] for j in range(5)}
    df = pd.concat([df, pd.DataFrame(dico)], axis=1)

    dico = {'age': [age],
            'height': [height],
            'weight': [weight],
            'gender': [gender],
            'C50p': [bis_param[0]],
            'C50r': [bis_param[1]],
            'gamma': [bis_param[2]],
            'beta': [bis_param[3]],
            'Emax': [bis_param[4]],
            'E0': [bis_param[5]]}
    pd_param = pd.concat([pd_param, pd.DataFrame(dico)], axis=0)

if phase == 'maintenance':
    df.to_csv("./Results_data/result_PID_maintenance_n=" + str(Number_of_patient) + '.csv')
elif phase == 'induction':
    df.to_csv("./Results_data/result_PID_induction_n=" + str(Number_of_patient) + '.csv')
else:
    df.to_csv("./Results_data/result_PID_total_n=" + str(Number_of_patient) + '.csv')

pd_param.hist(density=True)
