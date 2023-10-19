from loop import perform_simulation
import optuna
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import python_anesthesia_simulator as pas

# parameter of the simulation
phase = 'induction'
control_type = 'MEKF_NMPC'
Patient_number = 50


np.random.seed(0)
training_patient = np.random.randint(0, 500, size=16)


def small_obj(i: int, mekf_nmpc_param: list, output: str = 'IAE'):
    np.random.seed(i)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    df_results = perform_simulation([age, height, weight, gender], phase, control_type='EKF-NMPC',
                                    control_param=mekf_nmpc_param, random_bool=[True, True])
    if output == 'IAE':
        IAE = np.sum(np.abs(df_results['BIS'] - 50)*1)
        return IAE
    elif output == 'dataframe':
        return df_results
    else:
        return


# %% set observer parameters
study_petri = optuna.load_study(study_name="petri_final_3", storage="sqlite:///Results_data/petri_2.db")
Q_est = study_petri.best_params['Q'] * np.diag([0.1, 0.1, 0.05, 0.05, 1, 1, 10, 1, 1])
R_est = study_petri.best_params['R']
P0_est = 1e-3 * np.eye(9)

MEKF_param = [Q_est, R_est, P0_est]


def objective(trial):
    N = trial.suggest_int('N', 10, 30)
    R = trial.suggest_float('R', 1, 100, log=True) * np.diag([10, 1])
    Nu = N
    Q9 = trial.suggest_float('Q_9', 1.e-4, 10, log=True)
    Q_est = MEKF_param[0]
    Q_est[-1, -1] = Q9
    MEKF_param[0] = Q_est
    mekf_nmpc_param = MEKF_param + [N, Nu, R]
    local_cost = partial(small_obj, mekf_nmpc_param=mekf_nmpc_param)
    with mp.Pool(mp.cpu_count()-1) as p:
        r = list(p.imap(local_cost, training_patient))
    return max(r)


# %% Tuning of the controler
study = optuna.create_study(direction='minimize', study_name=f"EKF_MPC_{phase}_1",
                            storage='sqlite:///Results_data/tuning.db', load_if_exists=True)
study.optimize(objective, n_trials=10)

print(study.best_params)

MPC_param = [study.best_params['N'], study.best_params['N'], study.best_params['R']]
Q9 = study.best_params['Q_9']
Q_est = MEKF_param[0]
Q_est[-1, -1] = Q9
MEKF_param[0] = Q_est
mekf_nmpc_param = MEKF_param + MPC_param

# %% test on all patient

test_func = partial(small_obj, mekf_nmpc_param=mekf_nmpc_param, output='dataframe')

with mp.Pool(mp.cpu_count()-1) as p:
    res = list(tqdm(p.imap(test_func, range(Patient_number)), total=Patient_number, desc='Test MEKF MPC'))

print("Saving results...")
final_df = pd.DataFrame()

for i, df in enumerate(res):
    df.rename(columns={'Time': f"{i}_Time",
                       'BIS': f"{i}_BIS",
                       "u_propo": f"{i}u_propo",
                       "u_remi": f"{i}_u_remi",
                       "step_stime": f"{i}_step_time"}, inplace=True)

    final_df = pd.concat((final_df, df), axis=1)


final_df.to_csv(f"./Results_data/MEKF_MPC_{phase}_{Patient_number}.csv")

print("Done!")

# %% plot the results
linewidth = 0.5

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(final_df['0_Time']/60, final_df.loc[:, final_df.columns.str.endswith('BIS')], 'b', linewidth=linewidth)
plt.ylabel('BIS')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(final_df['0_Time']/60, final_df.loc[:, final_df.columns.str.endswith('u_remi')],
         'b', linewidth=linewidth)
plt.plot(final_df['0_Time']/60, final_df.loc[:, final_df.columns.str.endswith('u_propo')],
         'r', linewidth=linewidth)

plt.plot([], [], 'r', linewidth=linewidth, label='propofol')
plt.plot([], [], 'b', linewidth=linewidth, label='remifentanil')

plt.ylabel('Drug rates')
plt.xlabel('Time(min)')
plt.legend()
plt.grid()
plt.show()


print(f"mean computation time: {np.mean(final_df.loc[:, final_df.columns.str.endswith('step_time')])}s")
