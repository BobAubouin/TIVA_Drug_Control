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
Patient_number = 100


np.random.seed(0)
training_patient = np.random.randint(0, 500, size=16)


def small_obj(i: int, ekf_nmpc_param: list, output: str = 'IAE'):
    np.random.seed(i)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    df_results = perform_simulation([age, height, weight, gender], phase, control_type='EKF-NMPC',
                                    control_param=ekf_nmpc_param, random_bool=[True, True])
    if output == 'IAE':
        # IAE = np.sum((df_results['BIS'].loc[2*60//2:4*60//2] - 50)**2*(1+((df_results['BIS'].loc[2*60//2:4*60//2] - 50) < 0)) * 2, axis=0)
        mask = df_results['BIS'] > 50
        IAE = np.sum((df_results['BIS'] - 50)**2 * mask + (df_results['BIS'] - 50)**4 * (~mask), axis=0)
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

EKF_param = [Q_est, R_est, P0_est]


def objective(trial):
    N = 30
    R = trial.suggest_float('R', 100, 5.e3, log=True) * np.diag([2, 1])
    Nu = N
    Q9 = trial.suggest_float('Q_9', 1.e-3, 1.e-1, log=True)
    Q_est = EKF_param[0]
    Q_est[-1, -1] = Q9
    EKF_param[0] = Q_est
    ekf_nmpc_param = EKF_param + [N, Nu, R]
    local_cost = partial(small_obj, ekf_nmpc_param=ekf_nmpc_param)
    with mp.Pool(mp.cpu_count()-1) as p:
        r = list(p.imap(local_cost, training_patient))
    return max(r)


# %% Tuning of the controler
# search_space = {'R': np.logspace(-1,2,10), 'Q_9': np.logspace(-1,2,3)}
study = optuna.create_study(direction='minimize', study_name=f"EKF_MPC_{phase}_7",
                            storage='sqlite:///Results_data/tuning.db', load_if_exists=True)
                            # sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, n_trials=60, show_progress_bar=True)

print(study.best_params)

MPC_param = [30, 30, study.best_params['R'] * np.diag([4, 1])]
Q9 = study.best_params['Q_9']
Q_est = EKF_param[0]
Q_est[-1, -1] = Q9
EKF_param[0] = Q_est
mekf_nmpc_param = EKF_param + MPC_param

# %% test on all patient

test_func = partial(small_obj, ekf_nmpc_param=mekf_nmpc_param, output='dataframe')
patient_list = [i for i in range(Patient_number)]+training_patient.tolist()
with mp.Pool(mp.cpu_count()-1) as p:
    res = list(tqdm(p.map(test_func, patient_list), total=len(patient_list), desc='Test EKF'))

print("Saving results...")
final_df = pd.DataFrame()

for i in range(len(res)):
    df = res[i]
    patient_id = patient_list[i]
    df.rename(columns={'Time': f"{patient_id}_Time",
                       'BIS': f"{patient_id}_BIS",
                       "u_propo": f"{patient_id}_u_propo",
                       "u_remi": f"{patient_id}_u_remi",
                       "step_stime": f"{patient_id}_step_time"}, inplace=True)

    final_df = pd.concat((final_df, df), axis=1)


final_df.to_csv(f"./Results_data/EKF_NMPC_{phase}_{Patient_number}.csv")

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
plt.savefig(f"./Results_Images/EKF_NMPC_{phase}_{Patient_number}.png", dpi=300)
plt.show()


print(f"mean computation time: {np.mean(final_df.loc[:, final_df.columns.str.endswith('step_time')])}s")
