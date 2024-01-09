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
control_type = 'MHE_NMPC'
cost_choice = 'IAE'
Patient_number = 100


np.random.seed(0)
training_patient = np.random.randint(0, 500, size=16)

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
    if type == 'IAE':
        cost = np.sum((df['BIS'] - 50)**2, axis=0)
    elif type == 'IAE_biased':
        mask = df['BIS'] > 50
        cost = np.sum((df['BIS'] - 50)**3 * mask + (df['BIS'] - 50)**4 * (~mask), axis=0)
    elif type == 'TT':
        for i in range(len(df['BIS'])):
            if df['BIS'].iloc[i] < 60:
                break
        cost = (df['Time'].iloc[i] - 101)**2
    return cost

def small_obj(i: int, mhe_nmpc_param: list, output: str = 'IAE'):
    np.random.seed(i)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)
    print(i*0.1)
    df_results = perform_simulation([age, height, weight, gender], phase, control_type='MHE-NMPC',
                                    control_param=mhe_nmpc_param, random_bool=[True, True])
    if output == 'IAE':
        cost = compute_cost(df_results, cost_choice)
        return cost
    elif output == 'dataframe':
        print(i)
        return i, df_results
    else:
        return


# %% set observer parameters
# study_mhe = optuna.load_study(study_name="mhe_final_2", storage="sqlite:///Results_data/mhe.db")
gamma = 0.105  # study_mhe.best_params['eta']
theta = [gamma, 800, 100, 0.005]*4
theta[4] = gamma/100
theta[12] = gamma*10
theta[13] = 300
theta[15] = 0.05
Q_mhe = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
R_mhe = 0.016  # study_mhe.best_params['R']
N_mhe = 18  # study_mhe.best_params['N_mhe']
MHE_param = [Q_mhe, R_mhe, N_mhe, theta]


def objective(trial):
    N = 30
    R = trial.suggest_float('R', 10, 5.e3, log=True) * np.diag([4, 1])
    Nu = N
    theta_d = trial.suggest_float('theta_d', 1.e-4, 10, log=True)
    theta = MHE_param[3]
    theta[12] = gamma * theta_d
    MHE_param[3] = theta
    mhe_nmpc_param = MHE_param + [N, Nu, R]
    local_cost = partial(small_obj, mhe_nmpc_param=mhe_nmpc_param)
    with mp.Pool(mp.cpu_count()-1) as p:
        r = list(p.imap(local_cost, training_patient))
    return max(r)


# %% Tuning of the controler
study = optuna.create_study(direction='minimize', study_name=f"MHE_MPC_induction_7",
                            storage='sqlite:///Results_data/tuning.db', load_if_exists=True)
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(study.best_params)

MPC_param = [30, 30,  study.best_params['R'] * np.diag([4, 1])] # , 42
theta_d = study.best_params['theta_d']
theta = MHE_param[3]
theta[12] = gamma * theta_d
MHE_param[3] = theta
mhe_nmpc_param = MHE_param + MPC_param

# %% test on all patient
print("start simulation...")
test_func = partial(small_obj, mhe_nmpc_param=mhe_nmpc_param, output='dataframe')
patient_list = [i for i in range(Patient_number)]+training_patient.tolist()
print(patient_list)
print(mp.cpu_count())
with mp.Pool(mp.cpu_count()-1) as p:
    res = list(tqdm(p.map(test_func, patient_list), total=len(patient_list), desc=f"Test MHE {phase}"))

print("Saving results...")
final_df = pd.DataFrame()

for i in range(len(res)):

    df = res[i][1]
    patient_id = res[i][0]
    df.rename(columns={'Time': f"{patient_id}_Time",
                       'BIS': f"{patient_id}_BIS",
                       "u_propo": f"{patient_id}_u_propo",
                       "u_remi": f"{patient_id}_u_remi",
                       "step_stime": f"{patient_id}_step_time"}, inplace=True)

    final_df = pd.concat((final_df, df), axis=1)


final_df.to_csv(f"./Results_data/MHE_NMPC_{phase}_{Patient_number}.csv")

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
plt.savefig(f"./Results_Images/MHE_NMPC_{phase}_{Patient_number}.png", dpi=300)
plt.show()


print(f"mean computation time: {np.mean(final_df.loc[:, final_df.columns.str.endswith('step_time')])}s")
