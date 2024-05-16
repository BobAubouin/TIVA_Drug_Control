from functools import partial

import numpy as np
import pandas as pd
import optuna as opt
import os

from close_loop_anesth.loop import perform_simulation

np.random.seed(2)
training_patient = np.random.randint(1000, 1500, size=32)


def compute_cost(df: pd.DataFrame, type: str) -> float:
    """Compute the cost of the simulation.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of the simulation.
    type : str
        type of the cost. can be 'IAE' or 'TT'.

    Returns
    -------
    float
        cost of the simulation.
    """
    ts = df['Time'].iloc[1] - df['Time'].iloc[0]

    if type == 'IAE':
        cost = np.sum((df['BIS'] - 50)**2, axis=0) * ts
    elif type == 'IAE_biased':
        mask = df['BIS'] > 50
        cost = (np.sum((df['BIS'] - 50)**3 * mask + (df['BIS'] - 50)**4 * (~mask), axis=0)) * ts
    elif type == 'IAE_biased_normal':
        TIME_MAINTENANCE = 599
        bis_induction = df[df.Time < TIME_MAINTENANCE].BIS
        bis_maintenance = df[df.Time >= TIME_MAINTENANCE].BIS
        mask_induction = bis_induction > 50
        biased_cost = np.sum((bis_induction - 50)**3 * mask_induction +
                             (bis_induction - 50)**4 * (~mask_induction), axis=0)
        normal_cost = np.sum((bis_maintenance - 50)**4, axis=0)
        cost = (biased_cost + normal_cost) * ts
    elif type == 'IAE_biased_40_normal':
        TIME_MAINTENANCE = 599
        bis_induction = df[df.Time < TIME_MAINTENANCE].BIS
        bis_maintenance = df[df.Time >= TIME_MAINTENANCE].BIS
        mask_induction = bis_induction > 40
        biased_cost = np.sum((np.abs(bis_induction - 50))**2 * mask_induction +
                             (np.abs(bis_induction - 50))**2.3 * (~mask_induction), axis=0)
        normal_cost = np.sum(np.abs((bis_maintenance - 50))**2, axis=0)
        cost = (biased_cost + normal_cost) * ts
    elif type == 'TT':
        for i in range(len(df['BIS'])):
            if df['BIS'].iloc[i] < 60:
                break
        cost = (df['Time'].iloc[i] - 101)**2
    return cost


def random_simu(caseid: int,
                control_type: str,
                control_param: dict,
                estim_param: dict,
                output: str = 'IAE',
                phase: str = 'induction',
                cost_choice: str = 'cost'):
    np.random.seed(caseid)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    if control_type == 'PID':
        ts = 2
    else:
        ts = 2
    df_results = perform_simulation([age, height, weight, gender],
                                    phase,
                                    control_type=control_type,
                                    control_param=control_param,
                                    estim_param=estim_param,
                                    random_bool=[True, True],
                                    sampling_time=ts)
    if output == 'cost':
        cost = compute_cost(df_results, cost_choice)
        return cost
    elif output == 'dataframe':
        df_results.insert(0, 'caseid', caseid)
        return df_results
    else:
        return


def nominal_simu(caseid: int,
                 control_type: str,
                 control_param: dict,
                 estim_param: dict,
                 output: str = 'IAE',
                 phase: str = 'induction',
                 cost_choice: str = 'cost'):
    np.random.seed(caseid)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    df_results = perform_simulation([age, height, weight, gender],
                                    phase,
                                    control_type=control_type,
                                    control_param=control_param,
                                    estim_param=estim_param,
                                    random_bool=[False, False])
    if output == 'cost':
        cost = compute_cost(df_results, cost_choice)
        return cost
    elif output == 'dataframe':
        df_results.insert(0, 'caseid', caseid)
        return df_results
    else:
        return


def tunning_pid_function(trial, caseid, phase, cost_choice, ratio):
    control_param = {'Kp_1': trial.suggest_float('Kp_1', 1e-4, 1, log=True),
                     'Ti_1': trial.suggest_float('Ti_1', 100, 1500),
                     'Td_1': trial.suggest_float('Td_1', 0.1, 25),
                     'ratio': ratio}

    if phase == 'total':
        control_param['Kp_2'] = trial.suggest_float('Kp_2', 1e-4, 1)
        control_param['Ti_2'] = trial.suggest_float('Ti_2', 100, 1500)
        control_param['Td_2'] = trial.suggest_float('Td_2', 0.1, 25)

    estim_param = None

    cost = nominal_simu(caseid=caseid,
                        control_type='PID',
                        control_param=control_param,
                        estim_param=estim_param,
                        output='cost',
                        phase=phase,
                        cost_choice=cost_choice)

    return cost


def auto_tune_pid(caseid, phase, cost_choice, ratio, nb_of_step=1000):
    study_name = f'PID_{caseid}'
    study = opt.create_study(direction='minimize', study_name=study_name,
                             storage='sqlite:///data/optuna/tuning.db', load_if_exists=True)

    nb_trials = study.trials_dataframe().shape[0]
    nb_to_do = nb_of_step - nb_trials

    study.optimize(partial(tunning_pid_function, caseid=caseid, phase=phase, cost_choice=cost_choice, ratio=ratio),
                   n_trials=nb_to_do, show_progress_bar=True)

    return study.best_params
