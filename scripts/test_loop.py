from loop import perform_simulation
import scipy
import matplotlib.pyplot as plt
import numpy as np
import python_anesthesia_simulator as pas
import optuna
import time
import multiprocessing as mp
from functools import partial


study_petri = optuna.load_study(study_name="petri_final_3", storage="sqlite:///Results_data/petri_2.db")
Q_est = study_petri.best_params['Q'] * np.diag([0.1, 0.1, 0.05, 0.05, 1, 1, 10, 1, 10])
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


param = MEKF_param + [30, 30, 19 * np.diag([16, 1])]


study_mhe = optuna.load_study(study_name="mhe_final_2", storage="sqlite:///Results_data/mhe.db")
gamma = 0.105  # study_mhe.best_params['eta']
theta = [gamma, 0, 0, 0]*4
theta[4] = gamma/100
Q_mhe = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
R_mhe = 0.016  # study_mhe.best_params['R']
N_mhe = 18  # study_mhe.best_params['N_mhe']
param_mhe = [Q_mhe, R_mhe, N_mhe, theta] + [30, 30, 0.1 * np.diag([10, 1])]

param_ekf = [Q_est, R_est, P0_est, 30, 30, 19 * np.diag([10, 1])]

phase = 'induction'
control_type = 'EKF-NMPC'
Patient_number = 50
training_patient = np.random.randint(0, 500, size=3)


def small_obj(i: int, mhe_nmpc_param: list, output: str = 'IAE'):
    np.random.seed(i)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    start = time.perf_counter()
    df_results = perform_simulation([age, height, weight, gender],
                                    phase, control_type='MHE-NMPC',
                                    control_param=param_mhe, random_bool=[True, True])
    end = time.perf_counter()
    print(f" Time to perform {phase} phase : {end-start} s")
    if output == 'IAE':
        IAE = np.sum(np.abs(df_results['BIS'] - 50)*1)
        return IAE
    elif output == 'dataframe':
        return df_results
    else:
        return


local_cost = partial(small_obj, mhe_nmpc_param=param_mhe, output='dataframe')

start = time.perf_counter()
# with mp.Pool(mp.cpu_count()-1) as p:
#     r = list(p.map(local_cost, training_patient))
df = perform_simulation([30, 170, 70, 0], phase, control_type=control_type,
                        control_param=param_ekf, random_bool=[True, True])

end = time.perf_counter()
# df = r[0]
print(f" Time to perform {phase} phase : {end-start} s")
print(f" Mean step time : {df['step_time'].mean()} s")
plt.figure()
plt.plot(df['Time'], df['BIS'])
plt.plot(df['Time'], np.full(len(df['Time']), 50), '--')
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('BIS')
plt.title(f"{control_type} control")
plt.show()

plt.figure()
plt.plot(df['Time'], df['u_propo'], label='Propofol')
plt.plot(df['Time'], df['u_remi'], label='Remifentanil')
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Drug rate')
plt.title(f"{control_type} control")
plt.show()
