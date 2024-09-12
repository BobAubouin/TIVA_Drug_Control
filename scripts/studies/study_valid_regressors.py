from functools import partial
import multiprocessing as mp
import json
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8

import numpy as np
import pandas as pd
from tqdm import tqdm

from close_loop_anesth.experiments import random_simu_regressor
from create_param import load_mekf_param, load_mhe_param


# test_PID
# define the parameter of the sudy
control_type = 'PID'
cost_choice = 'IAE_biased_40_normal'
phase = 'total'
study_name = 'PID_training_ok'
patient_number = 1000

param_file = f'data/logs/{study_name}.json'
with open(param_file, 'r') as f:
    dict = json.load(f)

# run the best parameter on the test set
best_params = dict['best_params']
best_params['ratio'] = 2

print("Perform PID test...")
start = time.time()
test_func = partial(random_simu_regressor,
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
final_df.to_csv("./data/signals/pid_regressor.csv")

print("Done!")

# MEKF
# define the parameter of the sudy
control_type = 'MEKF_NMPC'
cost_choice = 'IAE_biased_40_normal'
phase = 'total'
study_name = 'MEKF_training_ok'
point_number = [5, 6, 6]
bool_non_linear = True
alpha = 1

param_file = f'data/logs/{study_name}.json'
with open(param_file, 'r') as f:
    dict = json.load(f)

# run the best parameter on the test set

best_params = dict['best_params']
control_param = {'R': best_params['R_mpc']*np.diag([4, 1]),
                 'N': best_params['N_mpc'],
                 'Nu': best_params['N_mpc'],
                 'bool_non_linear': bool_non_linear,
                 'R_maintenance': best_params['R_maintenance']*np.diag([4, 1])}

estim_param = load_mekf_param(point_number=point_number,
                              q=best_params['q'],
                              r=best_params['R'],
                              alpha=best_params['alpha'],
                              lambda_2=best_params['lambda_2'],
                              epsilon=best_params['epsilon'])

print("Perform MEKF test...")
start = time.time()
test_func = partial(random_simu_regressor,
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
final_df.to_csv("./data/signals/mekf_regressor.csv")

print("Done!")

# MHE
# define the parameter of the sudy
control_type = 'MHE_NMPC'
cost_choice = 'IAE_biased_40_normal'
phase = 'total'
study_name = 'MHE_training_ok'
vmax = 1e4
vmin = 0.01
bool_non_linear = True

param_file = f'data/logs/{study_name}.json'
with open(param_file, 'r') as f:
    dict = json.load(f)

# run the best parameter on the test set

best_params = dict['best_params']
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

print("Perform MHE test...")
start = time.time()
test_func = partial(random_simu_regressor,
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
final_df.to_csv("./data/signals/mhe_regressor.csv")

print("Done!")
