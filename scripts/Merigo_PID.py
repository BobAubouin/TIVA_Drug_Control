"""MISO PID control of anesthesia.

from L. Merigo, F. Padula, N. Latronico, M. Paltenghi, and A. Visioli, 
“Optimized PID control of propofol and remifentanil coadministration for general anesthesia,”
Communications in Nonlinear Science and Numerical Simulation
, vol. 72, pp. 194–212, Jun. 2019, doi: 10.1016/j.cnsns.2018.12.015.
"""

# %% import packages

# Standard import
import multiprocessing
from functools import partial

# Third party imports
import numpy as np
import pandas as pd
from bokeh.models import HoverTool
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
from pyswarm import pso
import casadi as cas

# Local imports
from controller import PID
import python_anesthesia_simulator as pas

#%% Define the simulation function

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
            BIS[i] = Bis
            MAP[i] = Map
            CO[i] = Co
            if i == N_simu - 1:
                break
            uP = PID_controller.one_step(Bis, BIS_target)
            uR = min(ur_max, max(0, uP * ratio))
            uP = min(up_max, max(0, uP))
            Up[i] = uP
            Ur[i] = uR

    elif style == 'total':
        N_simu = int(60 / ts) * 60
        BIS = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)
        uP = 0
        for i in range(N_simu):
            uR = min(ur_max, max(0, uP * ratio))
            uP = min(up_max, max(0, uP))
            Dist = pas.disturbances.compute_disturbances(i * ts, 'realistic')
            Bis, Co, Map, _ = George.one_step(uP, uR, dist=Dist, noise=False)

            BIS[i] = min(100, Bis)
            MAP[i] = Map[0, 0]
            CO[i] = Co[0, 0]
            Up[i] = uP
            Ur[i] = uR
            uP = PID_controller.one_step(Bis, BIS_target)

    elif style == 'maintenance':
        N_simu = 25 * 60  # 25 minutes
        BIS = np.zeros(N_simu)
        MAP = np.zeros(N_simu)
        CO = np.zeros(N_simu)
        Up = np.zeros(N_simu)
        Ur = np.zeros(N_simu)

        # find equilibrium input
        uP, uR = George.find_bis_equilibrium_with_ratio(BIS_target, ratio)
        #initialize the simulator with the equilibrium input
        George.initialized_at_given_input(u_propo= uP, u_remi = uR)
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
Patient_table = pd.read_csv('./Patient_table.csv')


# phase = 'maintenance'
phase = 'induction'


def one_simu(x, ratio, i):
    """Cost of one simulation, i is the patient index."""
    Patient_info = Patient_table.loc[i-1].to_numpy()[1:]
    iae, _, _ = simu(Patient_info, phase, [x[0], x[1], x[2], ratio])
    return iae


def cost(x, ratio):
    """Cost of the optimization, x is the vector of the PID controller.

    x = [Kp, Ti, Td]
    IAE is the maximum integrated absolut error over the patient population.
    """
    pool_obj = multiprocessing.Pool()
    func = partial(one_simu, x, ratio)
    IAE = pool_obj.map(func, range(1, 17))
    pool_obj.close()
    pool_obj.join()

    return max(IAE)


try:
    if phase == 'maintenance':
        param_opti = pd.read_csv('./optimal_parameters_PID_reject.csv')
    else:
        param_opti = pd.read_csv('./optimal_parameters_PID.csv')
except:
    param_opti = pd.DataFrame(columns=['ratio', 'Kp', 'Ti', 'Td'])
    for ratio in range(2, 3):
        def local_cost(x): return cost(x, ratio)
        lb = [1e-6, 100, 0.1]
        ub = [1, 300, 20]
        # Default: 100 particles as in the article
        xopt, fopt = pso(local_cost, lb, ub, debug=True, minfunc=1e-2)
        param_opti = pd.concat((param_opti, pd.DataFrame(
            {'ratio': ratio, 'Kp': xopt[0], 'Ti': xopt[1], 'Td': xopt[2]}, index=[0])), ignore_index=True)
        print(ratio)
    if phase == 'maintenance':
        param_opti.to_csv('./optimal_parameters_PID_reject.csv')
    else:
        param_opti.to_csv('./optimal_parameters_PID.csv')

# %%test on patient table
ts = 1
IAE_list = []
TT_list = []
p1 = figure(width=900, height=300)
p2 = figure(width=900, height=300)
p3 = figure(width=900, height=300)
for ratio in range(2, 3):
    print('ratio = ' + str(ratio))
    Kp = float(param_opti.loc[param_opti['ratio'] == ratio, 'Kp'])
    Ti = float(param_opti.loc[param_opti['ratio'] == ratio, 'Ti'])
    Td = float(param_opti.loc[param_opti['ratio'] == ratio, 'Td'])
    PID_param = [Kp, Ti, Td, ratio]
    for i in range(1, 14):
        Patient_info = Patient_table.loc[i-1].to_numpy()[1:]
        IAE, data, BIS_param = simu(Patient_info, phase, PID_param)
        source = pd.DataFrame(data=data[0], columns=['BIS'])
        source.insert(len(source.columns), "time", np.arange(0, len(data[0]))*ts/60)
        source.insert(len(source.columns), "Ce50_P", BIS_param[0])
        source.insert(len(source.columns), "Ce50_R", BIS_param[1])
        source.insert(len(source.columns), "gamma", BIS_param[2])
        source.insert(len(source.columns), "beta", BIS_param[3])
        source.insert(len(source.columns), "E0", BIS_param[4])
        source.insert(len(source.columns), "Emax", BIS_param[5])

        plot = p1.line(x='time', y='BIS', source=source, width=2)
        tooltips = [('Ce50_P', "@Ce50_P"), ('Ce50_R', "@Ce50_R"),
                    ('gamma', "@gamma"), ('beta', "@beta"),
                    ('E0', "@E0"), ('Emax', "@Emax")]
        p1.add_tools(HoverTool(renderers=[plot], tooltips=tooltips))
        p2.line(np.arange(0, len(data[0]))*ts/60,
                data[1], legend_label='MAP (mmgh)')
        p2.line(np.arange(0, len(data[0]))*ts/60, data[2]*10,
                legend_label='CO (cL/min)', line_color="#f46d43")
        p3.line(np.arange(0, len(data[3]))*ts/60, data[3],
                line_color="#006d43", legend_label='propofol (mg/min)', width=2)
        p3.line(np.arange(0, len(data[4]))*ts/60, data[4],
                line_color="#f46d43", legend_label='remifentanil (ng/min)', width=2)
        if phase == 'induction':
            TT, BIS_NADIR, ST10, ST20, US = pas.metrics.compute_control_metrics(np.arange(0, len(data[0]))*ts,
                                                                                data[0], phase=phase)
            TT_list.append(TT)
        else:
            TTp, BIS_NADIRp, TTn, BIS_NADIRn = pas.metrics.compute_control_metrics(np.arange(0, len(data[0]))*ts,
                                                                                   data[0], phase=phase)
            TT_list.append(TTp)
        IAE_list.append(IAE)
p1.title.text = 'BIS'
p3.title.text = 'Infusion rates'
p3.xaxis.axis_label = 'Time (min)'
grid = row(column(p3, p1, p2))

show(grid)

print("Mean IAE : " + str(np.mean(IAE_list)))
print("Mean TT : " + str(np.mean(TT_list)))
print("Min TT : " + str(np.min(TT_list)))
print("Max TT : " + str(np.max(TT_list)))

# %% Intra patient variability


# Simulation parameter
phase = 'induction'
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
for i in range(Number_of_patient):
    print(i)
    np.random.seed(i)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    Patient_info = [age, height, weight, gender] + [None] * 6
    IAE, data, bis_param = simu(Patient_info, phase, PID_param, random_PD=True, random_PK=True)
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
    df.to_csv("../Results_data/result_PID_maintenance_n=" + str(Number_of_patient) + '.csv')
else:
    df.to_csv("../Results_data/result_PID_n=" + str(Number_of_patient) + '.csv')

pd_param.hist(density=True)
