from time import perf_counter
import numpy as np
import pandas as pd
import pickle
import python_anesthesia_simulator as pas

from close_loop_anesth.pid import PID
from close_loop_anesth.mpc import NMPC_integrator_multi_shooting
from close_loop_anesth.ekf import EKF
from close_loop_anesth.mekf import MEKF
from close_loop_anesth.mhe import MHE
from close_loop_anesth.mekf_mhe import MEKF_MHE
# from close_loop_anesth.utils import custom_disturbance


filename = "./data/regressor/reg_KNeighborsRegressor_feat_All.pkl"
regressors = pickle.load(open(filename, 'rb'))
scaler = pickle.load(open("./data/regressor/scale_All.pkl", 'rb'))
regressor = regressors['BIS']


def regressor_bis(patient_sim: pas.Patient, hr: float):
    x_propo = patient_sim.propo_pk.x
    x_remi = patient_sim.remi_pk.x
    bmi = patient_sim.weight / patient_sim.height**2
    X = [[patient_sim.age,
          patient_sim.gender,
          patient_sim.height,
          patient_sim.weight,
          bmi,
          patient_sim.lbm,
          hr,
          x_remi[4],
          (x_propo[4] + x_propo[5])/2,
          x_propo[3],
          x_remi[3],
          x_propo[0],
          x_remi[0]
          ]]

    bis = regressor.predict(scaler.transform(X))
    return bis


def perform_simulation(Patient_info: list,
                       phase: str,
                       control_type: str,
                       control_param: dict,
                       estim_param: dict,
                       random_bool: list,
                       sampling_time: float = 1,
                       sampling_time_control: float = 5,
                       bool_noise: bool = True,
                       ) -> pd.DataFrame:
    """Run the simulation of the closed loop anesthesia system.

    The phase

    Parameters
    ----------
    Patient_info : list
        list of patient infos [age, height, weight, gender]
    phase : str
        "induction" or "total".
    control_type : str
        can be 'PID', MEKF_NMPC', 'EKF_NMPC', 'MHE_NMPC'
    control_param : dict
        dict of control parameter specific for each control type:
        - for PID [K, Ti, Td, ratio, K_2, Ti_2, Td_2], where _2 is used during maintenance.
        - for NMPC [N, Nu, R_nmpc]

    estim_param : dict
        dict of estimator parameter specific for each control type:
        - for PID [None]
        - for EKF_NMPC [Q, R, P0]
        - for MEKF_NMPC [Q, R, P0, grid_vector, eta0, lambda_1, lambda_2, nu, epsilon]
        - for MHE_NMPC [Q, R, N, theta]
        # - for MEKF-MHE_NMPC [Q_est, R_est, P0_est, grid_vector, eta0, design_param, Q_mhe, R_mhe, N_mhe, theta, switch_time]

    random_bool : list
        list of len 2, first index to add uncertainty in the PK model and second index to add uncertainty in the PD model.
    sampling_time : float, optional
        sampling time of the simulation. The default is 1.
    sampling_time_control : float, optional
        sampling time of the control. The default is 5.
    bool_noise : bool, optional
        add noise to the simulation. The default is True.

    Returns
    -------
    pd.DataFrame
        Dataframe with the results of the simulation.l
    """
    patient_simu = pas.Patient(Patient_info, save_data_bool=False,
                               random_PK=random_bool[0], random_PD=random_bool[1], model_bis='Bouillon', model_propo='Eleveld', model_remi='Eleveld', ts=sampling_time)

    hr = np.random.uniform(40, 80)
    # define input constraints
    bis_target = 50
    up_max = 6.67
    ur_max = 16.67

    # define controller
    if control_type == 'PID':
        control_param['ts'] = sampling_time
        ratio = control_param['ratio']
        control_param_induction = {'Kp': control_param['Kp_1'],
                                   'Ti': control_param['Ti_1'],
                                   'Td': control_param['Td_1'],
                                   'umin': 0,
                                   'umax': max(up_max, ur_max / ratio),
                                   }
        if phase == 'total':
            control_param_maintenance = {'Kp': control_param['Kp_2'],
                                         'Ti': control_param['Ti_2'],
                                         'Td': control_param['Td_2'],
                                         }

        controller = PID(**control_param_induction)
    else:
        control_param['ts'] = sampling_time_control
        estim_param['ts'] = sampling_time

        # get Nominal model from the patient info
        Patient_nominal_simu = pas.Patient(Patient_info,
                                           save_data_bool=False,
                                           random_PK=False,
                                           random_PD=False,
                                           model_bis='Bouillon',
                                           model_propo='Eleveld',
                                           model_remi='Eleveld'
                                           )

        Ap = Patient_nominal_simu.propo_pk.continuous_sys.A[:4, :4]
        Bp = Patient_nominal_simu.propo_pk.continuous_sys.B[:4]
        Ar = Patient_nominal_simu.remi_pk.continuous_sys.A[:4, :4]
        Br = Patient_nominal_simu.remi_pk.continuous_sys.B[:4]

        BIS_param_nominal = Patient_nominal_simu.hill_param

        A = np.block([[Ap, np.zeros((4, 4))], [np.zeros((4, 4)), Ar]])
        B = np.block([[Bp, np.zeros((4, 1))], [np.zeros((4, 1)), Br]])
        control_param['A'] = A
        control_param['B'] = B
        control_param['BIS_param'] = BIS_param_nominal
        control_param['umax'] = [up_max, ur_max]
        control_param['umin'] = [0, 0]
        control_param['bool_u_eq'] = True
        if 'R_maintenance' in control_param.keys():
            R_maintenance = control_param['R_maintenance']
            control_param.pop('R_maintenance')
        else:
            R_maintenance = None

        A_int = np.block([[Ap, np.zeros((4, 5))], [np.zeros((4, 4)), Ar, np.zeros((4, 1))], [np.zeros((1, 9))]])
        B_int = np.block([[Bp, np.zeros((4, 1))], [np.zeros((4, 1)), Br], [0, 0]])
        if control_type == 'EKF_NMPC' or control_type == 'MEKF_NMPC':
            estim_param['A'] = A_int
            estim_param['B'] = B_int
        else:
            estim_param['A'] = A
            estim_param['B'] = B
            estim_param['BIS_param'] = BIS_param_nominal

        if control_type == 'EKF_NMPC':
            estimator = EKF(**estim_param)
            controller = NMPC_integrator_multi_shooting(**control_param)
        elif control_type == 'MEKF_NMPC':
            estimator = MEKF(**estim_param)
            controller = NMPC_integrator_multi_shooting(**control_param)
        elif control_type == 'MHE_NMPC':
            estimator = MHE(**estim_param)
            controller = NMPC_integrator_multi_shooting(**control_param)
        elif control_type == 'MEKF_MHE_NMPC':
            estimator = MEKF_MHE(**estim_param)
            controller = NMPC_integrator_multi_shooting(**control_param)
        R_mpc = control_param['R']

    # define dataframe to return
    line_list = []
    u_propo, u_remi = 0, 0
    if phase == 'induction':
        N_simu = 30*60//sampling_time
    else:
        N_simu = 20*60//sampling_time

    for i in range(N_simu):
        if phase == 'induction':
            disturbance = [0, 0, 0]
        else:
            disturbance = pas.compute_disturbances(i*sampling_time, 'step', start_step=10*60, end_step=15*60)
            # disturbance = custom_disturbance(i*sampling_time)
            if i*sampling_time == 9*60 + 58:
                if control_type == 'PID':
                    controller.change_param(**control_param_maintenance)
                elif R_maintenance is not None:
                    R_mpc = R_maintenance

        bis, _, _, _ = patient_simu.one_step(u_propo, u_remi, noise=bool_noise, dist=disturbance)
        bis = regressor_bis(patient_simu, hr) + disturbance[0] + patient_simu.bis_noise[patient_simu.noise_index]

        if i == N_simu - 1:
            break
        start = perf_counter()
        if control_type == 'PID':
            u_temp = controller.one_step(bis[0], bis_target)
            if i*sampling_time % sampling_time_control == 0:
                u_remi = min(ur_max, max(0, u_temp * ratio))
                u_propo = min(up_max, max(0, u_temp))
        else:
            x_estimated, _ = estimator.one_step([u_propo, u_remi], bis[0])
            if control_type == 'EKF_NMPC':
                x_estimated = np.concatenate((x_estimated[:-1], BIS_param_nominal[:3], [x_estimated[-1]]))
            if i*sampling_time % sampling_time_control == 0:
                u_propo, u_remi = controller.one_step(x_estimated, bis_target, R_mpc)
        end = perf_counter()
        x = np.concatenate((patient_simu.propo_pk.x[:4], patient_simu.remi_pk.x[:4]))
        line = pd.DataFrame([[i*sampling_time, bis[0], u_propo, u_remi, end-start, x, hr]+Patient_info],
                            columns=['Time', 'BIS', 'u_propo', 'u_remi', 'step_time', 'x', 'heart_rate', 'age', 'height', 'weight', 'gender'])
        line_list.append(line)

        # if control_type == 'MHE_NMPC':
        #     df['u_propo_target'].loc[i] = controller.ueq[0]
        #     df['u_remi_target'].loc[i] = controller.ueq[1]
    df = pd.concat(line_list)
    return df
