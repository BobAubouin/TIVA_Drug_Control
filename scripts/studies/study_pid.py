from functools import partial
import multiprocessing as mp
import json
import time

import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm

from close_loop_anesth.experiments import random_simu, training_patient


# define the parameter of the sudy
control_type = 'PID'
cost_choice = 'IAE_biased_40_normal'
phase = 'total'
study_name = 'PID_sampling_time_5'
patient_number = 1000
nb_of_step = 1000


def study_pid(trial):
    ratio = 2
    control_param = {'Kp_1': trial.suggest_float('Kp_1', 1e-4, 1, log=True),
                     'Ti_1': trial.suggest_float('Ti_1', 100, 1500),
                     'Td_1': trial.suggest_float('Td_1', 0.01, 25, log=True),
                     'ratio': ratio}

    if phase == 'total':
        control_param['Kp_2'] = trial.suggest_float('Kp_2', 1e-4, 1, log=True)
        control_param['Ti_2'] = trial.suggest_float('Ti_2', 1, 600)
        control_param['Td_2'] = trial.suggest_float('Td_2', 0.01, 25, log=True)

    estim_param = None

    local_cost = partial(random_simu,
                         control_type=control_type,
                         control_param=control_param,
                         estim_param=estim_param,
                         output='cost',
                         phase=phase,
                         cost_choice=cost_choice)

    with mp.Pool(mp.cpu_count()) as p:
        r = list(p.map(local_cost, training_patient))
    return np.mean(r)


# create the optuna study
study = optuna.create_study(direction='minimize', study_name=study_name,
                            storage='sqlite:///data/optuna/tuning.db', load_if_exists=True)

nb_trials = study.trials_dataframe().shape[0]
nb_to_do = nb_of_step - nb_trials

study.optimize(study_pid, n_trials=nb_to_do, show_progress_bar=True)

print(study.best_params)

# save the parameter of the sudy as json file
dict = {'control_type': control_type,
        'cost_choice': cost_choice,
        'phase': phase,
        'filename': f'{study_name}.csv',
        'best_params': study.best_params,
        'best_value': study.best_value,
        'nb_of_step': nb_of_step,}
with open(f'data/logs/{study_name}.json', 'w') as f:
    json.dump(dict, f)


# run the best parameter on the test set
best_params = study.best_params
best_params['ratio'] = 2

start = time.time()
test_func = partial(random_simu,
                    control_type=control_type,
                    control_param=best_params,
                    estim_param=None,
                    output='dataframe',
                    phase=phase,
                    cost_choice=cost_choice)
patient_list = np.arange(patient_number)
with mp.Pool(mp.cpu_count()-1) as p:
    res = list(tqdm(p.imap(test_func, patient_list), total=len(patient_list), desc='Test PID'))

print(f"Simulation time: {time.time() - start:.2f} s")
# save the result of the test set
print("Saving results...")
final_df = pd.concat(res)
final_df.to_csv(f"./data/signals/{dict['filename']}")

print("Done!")
