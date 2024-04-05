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
from create_param import load_mhe_param

# define the parameter of the sudy
control_type = 'MHE_NMPC'
cost_choice = 'IAE_biased'
phase = 'induction'
study_name = 'MHE_NMPC_1'
patient_number = 500


def study_mhe(trial):
    R_mhe = trial.suggest_float('R', 1e-5, 1e-1, log=True)
    N_mhe = trial.suggest_int('N_mhe', 20, 30)
    vmax = 1e4
    vmin = 0.1
    N_mpc = trial.suggest_int('N_mpc', 20, 80)
    R_mpc = trial.suggest_float('R_mpc', 1e-2, 60)
    q = trial.suggest_float('q', 1e2, 1e6, log=True)

    control_param = {'R': R_mpc*np.diag([4, 1]),
                     'N': N_mpc,
                     'Nu': N_mpc}

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
    nb_cpu = min(mp.cpu_count()-1, len(training_patient))
    with mp.Pool(nb_cpu) as p:
        r = list(p.map(local_cost, training_patient))
    return max(r)


# create the optuna study
study = optuna.create_study(direction='minimize', study_name=study_name,
                            storage='sqlite:///data/optuna/tuning.db', load_if_exists=True)
study.optimize(study_mhe, n_trials=100, show_progress_bar=True)

print(study.best_params)

# save the parameter of the sudy as json file
dict = {'control_type': control_type,
        'cost_choice': cost_choice,
        'phase': phase,
        'filename': f'MHE_{phase}_{patient_number}',
        'best_params': study.best_params}
with open(f'data/logs/{study_name}.json', 'w') as f:
    json.dump(dict, f)


# run the best parameter on the test set
best_params = study.best_params
best_params['ratio'] = 2

control_param = {'R': best_params['R_mpc']*np.diag([4, 1]),
                 'N': best_params['N_mpc'],
                 'Nu': best_params['N_mpc']}
estim_param = load_mhe_param(
    vmax=best_params['vmax'],
    R=best_params['R'],
    N_mhe=best_params['N_mhe'])


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
    res = list(tqdm(p.imap(test_func, patient_list), total=len(patient_list), desc='Test MHE'))

print(f"Simulation time: {time.time() - start:.2f} s")
# save the result of the test set
print("Saving results...")
final_df = pd.concat(res)
final_df.to_csv(f"./data/signals/{dict['filename']}.csv")

print("Done!")
