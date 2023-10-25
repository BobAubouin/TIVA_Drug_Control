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
Patient_number = 500


np.random.seed(0)
training_patient = np.random.randint(0, 500, size=16)


def small_obj(i: int, mekf_nmpc_param: list, output: str = 'IAE'):
    np.random.seed(i)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    df_results = perform_simulation([age, height, weight, gender], phase, control_type='MEKF-NMPC',
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
lambda_1 = 1
lambda_2 = study_petri.best_params['lambda_2']
nu = 1.e-5
epsilon = study_petri.best_params['epsilon']
design_param = [lambda_1, lambda_2, nu, epsilon]

BIS_param_nominal = pas.BIS_model().hill_param

cv_c50p = 0.182
cv_c50r = 0.888
cv_gamma = 0.304
# estimation of log normal standard deviation
w_c50p = np.sqrt(np.log(1+cv_c50p**2))
w_c50r = np.sqrt(np.log(1+cv_c50r**2))
w_gamma = np.sqrt(np.log(1+cv_gamma**2))

c50p_list = BIS_param_nominal[0]*np.exp([-2*w_c50p, -w_c50p, -0.5*w_c50p, 0, w_c50p])  # , -w_c50p
c50r_list = BIS_param_nominal[1]*np.exp([-2*w_c50r, -w_c50r, -0.5*w_c50r, 0, w_c50r])
gamma_list = BIS_param_nominal[2]*np.exp([-2*w_gamma, -w_gamma, -0.5*w_gamma, 0, w_gamma])  #
# surrender list by Inf value
c50p_list = np.concatenate(([-np.Inf], c50p_list, [np.Inf]))
c50r_list = np.concatenate(([-np.Inf], c50r_list, [np.Inf]))
gamma_list = np.concatenate(([-np.Inf], gamma_list, [np.Inf]))


def get_probability(c50p_set: list, c50r_set: list, gamma_set: list, method: str) -> float:
    """_summary_

    Parameters
    ----------
    c50p_set : float
        c50p set.
    c50r_set : float
        c50r set.
    gamma_set : float
        gamma set.
    method : str
        method to compute the probability. can be 'proportional' or 'uniform'.

    Returns
    -------
    float
        propability of the parameter set.
    """
    if method == 'proportional':
        mean_c50p = 4.47
        mean_c50r = 19.3
        mean_gamma = 1.13
        # cv_c50p = 0.182
        # cv_c50r = 0.888
        # cv_gamma = 0.304
        w_c50p = np.sqrt(np.log(1+cv_c50p**2))
        w_c50r = np.sqrt(np.log(1+cv_c50r**2))
        w_gamma = np.sqrt(np.log(1+cv_gamma**2))
        c50p_normal = scipy.stats.lognorm(scale=mean_c50p, s=w_c50p)
        proba_c50p = c50p_normal.cdf(c50p_set[1]) - c50p_normal.cdf(c50p_set[0])

        c50r_normal = scipy.stats.lognorm(scale=mean_c50r, s=w_c50r)
        proba_c50r = c50r_normal.cdf(c50r_set[1]) - c50r_normal.cdf(c50r_set[0])

        gamma_normal = scipy.stats.lognorm(scale=mean_gamma, s=w_gamma)
        proba_gamma = gamma_normal.cdf(gamma_set[1]) - gamma_normal.cdf(gamma_set[0])

        proba = proba_c50p * proba_c50r * proba_gamma
    elif method == 'uniform':
        proba = 1/(len(c50p_list))/(len(c50r_list))/(len(gamma_list))
    return proba


grid_vector = []
eta0 = []
proba = []
alpha = 10
for i, c50p in enumerate(c50p_list[1:-1]):
    for j, c50r in enumerate(c50r_list[1:-1]):
        for k, gamma in enumerate(gamma_list[1:-1]):
            grid_vector.append([c50p, c50r, gamma]+BIS_param_nominal[3:])
            c50p_set = [np.mean([c50p_list[i], c50p]),
                        np.mean([c50p_list[i+2], c50p])]

            c50r_set = [np.mean([c50r_list[j], c50r]),
                        np.mean([c50r_list[j+2], c50r])]

            gamma_set = [np.mean([gamma_list[k], gamma]),
                         np.mean([gamma_list[k+2], gamma])]

            eta0.append(alpha*(1-get_probability(c50p_set, c50r_set, gamma_set, 'proportional')))


MEKF_param = [Q_est, R_est, P0_est, grid_vector, eta0, design_param]


def objective(trial):
    N = 30
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
study = optuna.create_study(direction='minimize', study_name=f"MEKF_MPC_{phase}_2",
                            storage='sqlite:///Results_data/tuning.db', load_if_exists=True)
study.optimize(objective, n_trials=200)

print(study.best_params)

MPC_param = [study.best_params['N'], study.best_params['N'], study.best_params['R']* np.diag([10, 1])]
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
plt.savefig(f"./Results_Images/MEKF_MPC_{phase}_{Patient_number}.png", dpi=300)

plt.show()


print(f"mean computation time: {np.mean(final_df.loc[:, final_df.columns.str.endswith('step_time')])}s")
