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
control_type = 'MEKF_MHE_NMPC'
Patient_number = 500
cost_choice = 'TT'


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
    df_results = perform_simulation([age, height, weight, gender], phase, control_type='MEKF-MHE-NMPC',
                                    control_param=mhe_nmpc_param, random_bool=[True, True])
    if output == 'IAE':
        cost = compute_cost(df_results, cost_choice)
        return cost
    elif output == 'dataframe':
        return i, df_results
    else:
        return


# %% set observer parameters
# study_mhe = optuna.load_study(study_name="mhe_final_2", storage="sqlite:///Results_data/mhe.db")
gamma = 0.105  # study_mhe.best_params['eta']
theta = [gamma, 800, 100, 0.005]*4
theta[4] = gamma/100
theta[12] = gamma*0.08
theta[13] = 300
theta[15] = 0.05
Q_mhe = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
R_mhe = 0.00023678046027518807  # study_mhe.best_params['R']
N_mhe = 26  # study_mhe.best_params['N_mhe']
MHE_param = [Q_mhe, R_mhe, N_mhe, theta]

study_petri = optuna.load_study(study_name="petri_final_3", storage="sqlite:///Results_data/petri_2.db")
Q_est = study_petri.best_params['Q'] * np.diag([0.1, 0.1, 0.05, 0.05, 1, 1, 10, 1, 0.00001])
R_est = study_petri.best_params['R']
P0_est = 1e-3 * np.eye(9)
P0_est[8, 8] = 1e-6
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


parem_mekf_mhe = [Q_est, R_est, P0_est, grid_vector, eta0, design_param,
                  Q_mhe, R_mhe, N_mhe, theta, 180]


def objective(trial):
    N = 30
    R = trial.suggest_float('R', 5, 100, log=True) * np.diag([4, 1])  # * np.diag([4, 1])
    Nu = N
    t_switch = trial.suggest_int('t_switch', 20, 240)
    parem_mekf_mhe[-1] = t_switch
    Qest_d = trial.suggest_float('theta_d', 1.e-6, 10, log=True)
    Qest = parem_mekf_mhe[0]
    Qest[-1, -1] = Qest_d
    parem_mekf_mhe[0] = Qest

    mhe_nmpc_param = parem_mekf_mhe + [N, Nu, R]
    local_cost = partial(small_obj, mhe_nmpc_param=mhe_nmpc_param)
    with mp.Pool(mp.cpu_count()-1) as p:
        r = list(p.imap(local_cost, training_patient))
    return np.mean(r)


if __name__ == '__main__':
    # %% Tuning of the controler
    study = optuna.create_study(direction='minimize', study_name=f"MMMPC_induction_5",
                                storage='sqlite:///Results_data/tuning.db', load_if_exists=True)
    # study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(study.best_params)

    MPC_param = [30, 30, study.best_params['R'] * np.diag([4, 1])]  # , 39, study.best_params['R']
    t_switch = study.best_params['t_switch']
    parem_mekf_mhe[-1] = t_switch
    Qest_d = study.best_params['theta_d']
    Qest = parem_mekf_mhe[0]
    Qest[-1, -1] = Qest_d
    parem_mekf_mhe[0] = Qest
    mhe_nmpc_param = parem_mekf_mhe + MPC_param

    # %% test on all patient
    print("start simulation...")
    test_func = partial(small_obj, mhe_nmpc_param=mhe_nmpc_param, output='dataframe')
    patient_list = [i for i in range(Patient_number)]+training_patient.tolist()
    print(patient_list)
    print(mp.cpu_count())
    with mp.Pool(mp.cpu_count()//2) as p:
        res = list(tqdm(p.map(test_func, patient_list), total=len(patient_list), desc='Test MEKF_MHE'))

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

    final_df.to_csv(f"./Results_data/MEKF_MHE_NMPC_{phase}_{Patient_number}.csv")

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
    plt.savefig(f"./Results_Images/MEKF_MHE_NMPC_{phase}_{Patient_number}.png", dpi=300)
    plt.show()

    print(f"mean computation time: {np.mean(final_df.loc[:, final_df.columns.str.endswith('step_time')])}s")
