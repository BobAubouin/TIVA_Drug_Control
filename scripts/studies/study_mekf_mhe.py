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
from create_param import load_mekf_param, load_mhe_param

# define the parameter of the sudy
control_type = 'MEKF_MHE_NMPC'
cost_choice = 'IAE'
phase = 'total'
study_name = 'MEKF_MHE_mixt_test'
patient_number = 10
mekf_param_study = 'MEKF_mixt'
mhe_param_study = 'MHE_mixt'
bool_non_linear = True
nb_of_step = 2


# load mekf_param
with open(f'data/logs/{mekf_param_study}.json', 'r') as f:
    mekf_param = json.load(f).get('best_params')

mekf_param = load_mekf_param(point_number=mekf_param['point_number'],
                             q=mekf_param['q'],
                             r=mekf_param['R'],
                             alpha=mekf_param['alpha'],
                             lambda_2=mekf_param['lambda_2'],
                             epsilon=mekf_param['epsilon'])

# load mhe_param
with open(f'data/logs/{mhe_param_study}.json', 'r') as f:
    mhe_param = json.load(f).get('best_params')

mhe_param = load_mhe_param(vmax=mhe_param['vmax'],
                           vmin=mhe_param['vmin'],
                           R=mhe_param['R'],
                           N_mhe=mhe_param['N_mhe'],
                           q=mhe_param['q'])

# rename dictionary key
mekf_param['R_mekf'] = mekf_param.pop('R')
mekf_param['Q_mekf'] = mekf_param.pop('Q')
mekf_param['P0_mekf'] = mekf_param.pop('P0')

mhe_param['R_mhe'] = mhe_param.pop('R')
mhe_param['Q_mhe'] = mhe_param.pop('Q')
mhe_param['P_mhe'] = mhe_param.pop('P')

estim_param = {**mekf_param, **mhe_param}


def study_mekf_mhe(trial):
    switch_time = trial.suggest_int('switch_time', 1, 600)
    N_mpc = trial.suggest_int('N_mpc', 20, 80)
    R_mpc = trial.suggest_float('R_mpc', 1e-2, 60)

    control_param = {'R': R_mpc*np.diag([4, 1]),
                     'N': N_mpc,
                     'Nu': N_mpc,
                     'bool_non_linear': bool_non_linear}

    estim_param['switch_time'] = switch_time

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
study.optimize(study_mekf_mhe, n_trials=nb_to_do, show_progress_bar=True)

print(study.best_params)

best_params = study.best_params

# save the parameter of the sudy as json file
dict = {'control_type': control_type,
        'cost_choice': cost_choice,
        'phase': phase,
        'filename': f'{study_name}.csv',
        'mekf_param_study': mekf_param_study,
        'mhe_param': mhe_param_study,
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

estim_param = {**mekf_param, **mhe_param}
estim_param['switch_time'] = best_params['switch_time']


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
    res = list(tqdm(p.imap(test_func, patient_list), total=len(patient_list), desc='Test MEKF-MHE'))

print(f"Simulation time: {time.time() - start:.2f} s")
# save the result of the test set
print("Saving results...")
final_df = pd.concat(res)
final_df.to_csv(f"./data/signals/{dict['filename']}")

print("Done!")
