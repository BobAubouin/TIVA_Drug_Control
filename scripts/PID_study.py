from loop import perform_simulation
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# parameter of the simulation
phase = 'induction'
control_type = 'PID'
Patient_number = 500


np.random.seed(0)
training_patient = np.random.randint(0, 500, size=16)


def small_obj(i: int, pid_param: list, output: str = 'IAE'):
    np.random.seed(i)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    df_results = perform_simulation([age, height, weight, gender], phase, control_type='PID',
                                    control_param=pid_param, random_bool=[True, True])
    if output == 'IAE':
        IAE = np.sum(np.abs(df_results['BIS'] - 50)*1)
        return IAE
    elif output == 'dataframe':
        return df_results
    else:
        return


def objective(trial):
    K = trial.suggest_float('K', 1.e-3, 1)
    Ti = trial.suggest_float('Ti', 100, 600)
    Td = trial.suggest_float('Td', 0.1, 20)
    ratio = 2
    pid_param = [K, Ti, Td, ratio]
    local_cost = partial(small_obj, pid_param=pid_param)
    with mp.Pool(mp.cpu_count()-1) as p:
        r = list(p.imap(local_cost, training_patient))
    return max(r)


# %% Tuning of the controler
study = optuna.create_study(direction='minimize', study_name=f"PID_{phase}_1",
                            storage='sqlite:///Results_data/tuning.db', load_if_exists=True)
study.optimize(objective, n_trials=1000)

print(study.best_params)

pid_param = [study.best_params['K'], study.best_params['Ti'], study.best_params['Td'], 2]

# %% test on all patient

test_func = partial(small_obj, pid_param=pid_param, output='dataframe')

with mp.Pool(mp.cpu_count()-1) as p:
    res = list(tqdm(p.imap(test_func, range(Patient_number)), total=Patient_number, desc='Test PID'))

print("Saving results...")
final_df = pd.DataFrame()

for i, df in enumerate(res):
    df.rename(columns={'Time': f"{i}_Time",
                       'BIS': f"{i}_BIS",
                       "u_propo": f"{i}u_propo",
                       "u_remi": f"{i}_u_remi",
                       "step_stime": f"{i}_step_time"}, inplace=True)

    final_df = pd.concat((final_df, df), axis=1)


final_df.to_csv(f"./Results_data/PID_{phase}_{Patient_number}.csv")

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
plt.savefig(f"./Results_Images/PID_{phase}_{Patient_number}.png", dpi=300)
plt.show()

print(f"mean computation time: {np.mean(final_df.loc[:, final_df.columns.str.endswith('step_time')])}s")
