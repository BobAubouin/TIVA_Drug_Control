from functools import partial
import multiprocessing as mp
import json
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8

import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm

from close_loop_anesth.experiments import random_simu, training_patient_index
from create_param import load_mhe_param

# define the parameter of the sudy
control_type = 'MHE_NMPC'
cost_choice = 'IAE_biased_40_normal'
phase = 'total'
study_name = 'MHE_training_ok'
patient_number = 1000
vmax = 1e4
vmin = 0.01
bool_non_linear = True
nb_of_step = 200


def study_mhe(trial):
    R_mhe = trial.suggest_float('R', 1e-5, 1e-1, log=True)
    N_mhe = trial.suggest_int('N_mhe', 10, 30)
    N_mpc = trial.suggest_int('N_mpc', 10, 80)
    R_mpc = trial.suggest_float('R_mpc', 10, 500, log=True)
    R_maintenance = trial.suggest_float('R_maintenance', 1e-1, 1e2, log=True)
    q = 1e3  # trial.suggest_float('q', 1e2, 1e6, log=True)

    control_param = {'R': R_mpc*np.diag([4, 1]),
                     'N': N_mpc,
                     'Nu': N_mpc,
                     'bool_non_linear': bool_non_linear,
                     'R_maintenance': R_maintenance*np.diag([4, 1])}

    if not bool_non_linear:
        terminal_factor = trial.suggest_float('terminal_cost_factor', 1e-1, 1e3, log=True)
        control_param['terminal_cost_factor'] = terminal_factor

    estim_param = load_mhe_param(
        vmax=vmax,
        vmin=vmin,
        R=R_mhe,
        N_mhe=N_mhe,
        q=q)

    local_cost = partial(random_simu,
                         control_type=control_type,
                         control_param=control_param,
                         estim_param=estim_param,
                         output='cost',
                         phase=phase,
                         cost_choice=cost_choice)
    nb_cpu = min(mp.cpu_count(), len(training_patient_index))
    with mp.Pool(nb_cpu) as p:
        r = list(p.map(local_cost, training_patient_index))
    return np.mean(r)


# create the optuna study
study = optuna.create_study(direction='minimize', study_name=study_name,
                            storage='sqlite:///data/optuna/tuning.db', load_if_exists=True)
nb_to_do = nb_of_step - study.trials_dataframe().shape[0]

study.optimize(study_mhe, n_trials=nb_to_do, show_progress_bar=True)

print(study.best_params)

best_params = study.best_params
best_params['vmax'] = vmax
best_params['vmin'] = vmin
best_params['bool_non_linear'] = bool_non_linear
best_params['q'] = 1e3

# save the parameter of the sudy as json file
dict = {'control_type': control_type,
        'cost_choice': cost_choice,
        'phase': phase,
        'filename': f'{study_name}.csv',
        'best_params': best_params,
        'best_value': study.best_value,
        'nb_of_step': nb_of_step, }
with open(f'data/logs/{study_name}.json', 'w') as f:
    json.dump(dict, f)


# run the best parameter on the test set


control_param = {'R': best_params['R_mpc']*np.diag([4, 1]),
                 'N': best_params['N_mpc'],
                 'Nu': best_params['N_mpc'],
                 'bool_non_linear': bool_non_linear,
                 'R_maintenance': best_params['R_maintenance']*np.diag([4, 1])}

if not bool_non_linear:
    control_param['terminal_cost_factor'] = best_params['terminal_cost_factor']

estim_param = load_mhe_param(
    vmax=best_params['vmax'],
    vmin=best_params['vmin'],
    R=best_params['R'],
    N_mhe=best_params['N_mhe'],
    q=best_params['q'])


start = time.time()
test_func = partial(random_simu,
                    control_type=control_type,
                    control_param=control_param,
                    estim_param=estim_param,
                    output='dataframe',
                    phase=phase,
                    cost_choice=cost_choice)
patient_list = np.arange(patient_number)
with mp.Pool(mp.cpu_count()) as p:
    res = list(tqdm(p.imap(test_func, patient_list), total=len(patient_list), desc='Test MHE'))

print(f"Simulation time: {time.time() - start:.2f} s")
# save the result of the test set
print("Saving results...")
final_df = pd.concat(res)
final_df.to_csv(f"./data/signals/{dict['filename']}")

print("Done!")
