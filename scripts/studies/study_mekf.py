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

from close_loop_anesth.experiments import random_simu, training_patient
from create_param import load_mekf_param

# define the parameter of the sudy
control_type = 'MEKF_NMPC'
cost_choice = 'IAE_biased_normal'
phase = 'total'
study_name = 'MEKF_mixt'
patient_number = 500
point_number = [5, 6, 6]
bool_non_linear = True
nb_of_step = 100


def study_mekf(trial):
    r = trial.suggest_float('R', 1e-3, 1e4, log=True)
    epsilon = trial.suggest_float('epsilon', 0.1, 0.99)
    lambda_2 = trial.suggest_float('lambda_2', 1e2, 1e2, log=True)
    alpha = trial.suggest_float('alpha', 1e-1, 100, log=True)
    q = trial.suggest_float('q', 1e-3, 1e6, log=True)
    N_mpc = trial.suggest_int('N_mpc', 20, 80)
    R_mpc = trial.suggest_float('R_mpc', 1e-2, 60)

    control_param = {'R': R_mpc*np.diag([4, 1]),
                     'N': N_mpc,
                     'Nu': N_mpc,
                     'bool_non_linear': bool_non_linear}

    estim_param = load_mekf_param(point_number=point_number,
                                  q=q,
                                  r=r,
                                  alpha=alpha,
                                  lambda_2=lambda_2,
                                  epsilon=epsilon)

    local_cost = partial(random_simu,
                         control_type=control_type,
                         control_param=control_param,
                         estim_param=estim_param,
                         output='cost',
                         phase=phase,
                         cost_choice=cost_choice)
    nb_cpu = min(mp.cpu_count()-1, len(training_patient))
    with mp.Pool(nb_cpu) as p:
        r = list(p.map(local_cost, training_patient))
    return np.mean(r)


# create the optuna study
study = optuna.create_study(direction='minimize', study_name=study_name,
                            storage='sqlite:///data/optuna/tuning.db', load_if_exists=True)
# get number of trials
nb_trials = study.trials_dataframe().shape[0]
nb_to_do = nb_of_step - nb_trials

study.optimize(study_mekf, n_trials=nb_to_do, show_progress_bar=True)

print(study.best_params)

best_params = study.best_params
best_params['point_number'] = point_number

# save the parameter of the sudy as json file
dict = {'control_type': control_type,
        'cost_choice': cost_choice,
        'phase': phase,
        'filename': f'{study_name}.csv',
        'best_params': best_params,
        'best_score': study.best_value,
        'nb_of_step': nb_of_step, }
with open(f'data/logs/{study_name}.json', 'w') as f:
    json.dump(dict, f)


# run the best parameter on the test set


control_param = {'R': best_params['R_mpc']*np.diag([4, 1]),
                 'N': best_params['N_mpc'],
                 'Nu': best_params['N_mpc'],
                 'bool_non_linear': bool_non_linear}

estim_param = load_mekf_param(point_number=point_number,
                              q=best_params['q'],
                              r=best_params['R'],
                              alpha=best_params['alpha'],
                              lambda_2=best_params['lambda_2'],
                              epsilon=best_params['epsilon'])


start = time.time()
test_func = partial(random_simu,
                    control_type=control_type,
                    control_param=control_param,
                    estim_param=estim_param,
                    output='dataframe',
                    phase=phase,
                    cost_choice=cost_choice)

patient_list = np.arange(patient_number)
with mp.Pool(mp.cpu_count()-1) as p:
    res = list(tqdm(p.imap(test_func, patient_list), total=len(patient_list), desc='Test MEKF'))

print(f"Simulation time: {time.time() - start:.2f} s")
# save the result of the test set
print("Saving results...")
final_df = pd.concat(res)
final_df.to_csv(f"./data/signals/{dict['filename']}")

print("Done!")
