from time import perf_counter
import numpy as np
import pandas as pd
import python_anesthesia_simulator as pas
from estimators import MEKF, MHE_integrator, EKF_integrator_new, MEKF_MHE
from controller import PID, NMPC_integrator


def perform_simulation(Patient_info: list, phase: str, control_type: str, control_param: list, random_bool: list) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    Patient_info : list
        list of patient infos [age, height, weight, gender]
    phase : str
        "induction" or "total".
    control_type : str
        can be 'PID', MEKF-NMPC', 'EKF-NMPC', 'MHE-NMPC'
    control_param : list
        list of control parameter specific for each control type: 
        - for PID [K, Ti, Td, ratio]
        - for EKF-NMPC [Q_ekf, R_ekf, P0_ekf, N, Nu, R_nmpc]
        - for MEKF-NMPC [Q_est, R_est, P0_est, grid_vector, eta0, design_param, N, Nu, R_nmpc]
        - for MHE-NMPC [Q_mhe, R_mhe, N_mhe, theta, N, Nu, R_nmpc]
        - for MEKF-MHE-NMPC [Q_est, R_est, P0_est, grid_vector, eta0, design_param, Q_mhe, R_mhe, N_mhe, theta, switch_time, N, Nu, R_nmpc]
    random_bool : list
        list of len 2, first index to add uncertainty in the PK model and second index to add uncertainty in the PD model.

    Returns
    -------
    pd.DataFrame
        Dataframe with the results of the simulation.l
    """
    # define sampling time
    ts = 2

    patient_simu = pas.Patient(Patient_info, save_data_bool=False,
                               random_PK=random_bool[0], random_PD=random_bool[1], model_bis='Bouillon', model_propo='Eleveld', model_remi='Eleveld', ts=ts)

    # define input constraints
    bis_target = 50
    up_max = 6.67
    ur_max = 16.67

    # define controller
    if control_type == 'PID':
        controller = PID(Kp=control_param[0], Ti=control_param[1], Td=control_param[2],
                         N=5, Ts=ts, umax=max(up_max, ur_max / control_param[3]), umin=0)
    else:
        # get Nominal model from the patient info
        Patient_nominal_simu = pas.Patient(Patient_info, save_data_bool=False, random_PK=False,
                                           random_PD=False, model_bis='Bouillon', model_propo='Eleveld', model_remi='Eleveld')
        Ap = Patient_nominal_simu.propo_pk.continuous_sys.A[:4, :4]
        Bp = Patient_nominal_simu.propo_pk.continuous_sys.B[:4]
        Ar = Patient_nominal_simu.remi_pk.continuous_sys.A[:4, :4]
        Br = Patient_nominal_simu.remi_pk.continuous_sys.B[:4]

        BIS_param_nominal = Patient_nominal_simu.hill_param

        A = np.block([[Ap, np.zeros((4, 5))], [np.zeros((4, 4)), Ar, np.zeros((4, 1))], [np.zeros((1, 9))]])
        B = np.block([[Bp, np.zeros((4, 1))], [np.zeros((4, 1)), Br], [0, 0]])

        A_mhe = np.block([[Ap, np.zeros((4, 4))], [np.zeros((4, 4)), Ar]])
        B_mhe = np.block([[Bp, np.zeros((4, 1))], [np.zeros((4, 1)), Br]])

        if control_type == 'EKF-NMPC':
            estimator = EKF_integrator_new(A, B, BIS_param_nominal, ts,
                                           Q=control_param[0], R=control_param[1], P0=control_param[2])
            controller = NMPC_integrator(A, B, BIS_param_nominal, ts, N=control_param[3], Nu=control_param[4],
                                         R=control_param[5], umax=[up_max, ur_max], umin=[0, 0], bool_u_eq=True)
        elif control_type == 'MEKF-NMPC':
            estimator = MEKF(A, B, ts=ts, Q=control_param[0], R=control_param[1], P0=control_param[2],
                             grid_vector=control_param[3], eta0=control_param[4], design_param=control_param[5])
            controller = NMPC_integrator(A, B, BIS_param_nominal, ts, N=control_param[6], Nu=control_param[7],
                                         R=control_param[8], umax=[up_max, ur_max], umin=[0, 0], bool_u_eq=True)
        elif control_type == 'MHE-NMPC':
            estimator = MHE_integrator(
                A_mhe, B_mhe, BIS_param_nominal, ts, Q=control_param[0], R=control_param[1], N_MHE=control_param[2], theta=control_param[3])
            controller = NMPC_integrator(A, B, BIS_param_nominal, ts, N=control_param[4], Nu=control_param[5],
                                         R=control_param[6], umax=[up_max, ur_max], umin=[0, 0], bool_u_eq=True)
        elif control_type == 'MEKF-MHE-NMPC':
            estimator = MEKF_MHE(
                A, B, BIS_param_nominal, A_mhe, B_mhe, ts=ts, mekf_param=control_param[0:6], mhe_param=control_param[6:10], switch_time=control_param[10])
            controller = NMPC_integrator(A, B, BIS_param_nominal, ts, N=control_param[11], Nu=control_param[12],
                                         R=control_param[13], umax=[up_max, ur_max], umin=[0, 0], bool_u_eq=True)

    # define dataframe to return
    df = pd.DataFrame(columns=['Time', 'BIS', 'u_propo', 'u_remi', 'step_time'])
    u_propo, u_remi = 0, 0
    if phase == 'induction':
        N_simu = 10*60//ts
        bool_noise = True
    else:
        N_simu = 20*60//ts
        bool_noise = True

    for i in range(N_simu):
        if phase == 'induction':
            disturbance = [0, 0, 0]
        else:
            disturbance = pas.compute_disturbances(i*ts, 'step', start_step=10*60, end_step=15*60)

        bis, _, _, _ = patient_simu.one_step(u_propo, u_remi, noise=bool_noise, dist=disturbance)

        if i == N_simu - 1:
            break
        start = perf_counter()
        if control_type == 'PID':

            u_propo = controller.one_step(bis[0], bis_target)
            u_remi = min(ur_max, max(0, u_propo * control_param[3]))
            u_propo = min(up_max, max(0, u_propo))
        else:
            x_estimated, _ = estimator.one_step([u_propo, u_remi], bis[0])
            if control_type == 'EKF-NMPC':
                x_estimated = np.concatenate((x_estimated, BIS_param_nominal[:3]))
            u_propo, u_remi = controller.one_step(x_estimated, bis_target)
        end = perf_counter()
        line = pd.DataFrame([[i*ts, bis[0], u_propo, u_remi, end-start]],
                            columns=['Time', 'BIS', 'u_propo', 'u_remi', 'step_time'])
        df = pd.concat((df, line))
        # if control_type == 'MHE-NMPC':
        #     df['u_propo_target'].loc[i] = controller.ueq[0]
        #     df['u_remi_target'].loc[i] = controller.ueq[1]
    return df
