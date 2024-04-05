import numpy as np
import python_anesthesia_simulator as pas
import optuna
import scipy


def get_probability(c50p_set: list,
                    c50r_set: list,
                    gamma_set: list,
                    method: str,
                    normals: list = None,
                    lists: list = None) -> float:
    """Compute the probability of the parameter set.

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
    normals : list
        list of normal distribution for each parameter.
    lists : list
        list of all parameter values.

    Returns
    -------
    float
        propability of the parameter set.
    """
    if method == 'proportional':
        c50p_normal, c50r_normal, gamma_normal = normals
        proba_c50p = c50p_normal.cdf(c50p_set[1]) - c50p_normal.cdf(c50p_set[0])

        proba_c50r = c50r_normal.cdf(c50r_set[1]) - c50r_normal.cdf(c50r_set[0])

        proba_gamma = gamma_normal.cdf(gamma_set[1]) - gamma_normal.cdf(gamma_set[0])

        proba = proba_c50p * proba_c50r * proba_gamma
    elif method == 'uniform':
        c50p_list, c50r_list, gamma_list = lists
        proba = 1/(len(c50p_list))/(len(c50r_list))/(len(gamma_list))
    return proba


def init_proba(alpha, lists, BIS_param_nominal, normals):
    c50p_list, c50r_list, gamma_list = lists
    grid_vector = []
    eta0 = []
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

                eta0.append(alpha*(1-get_probability(c50p_set, c50r_set, gamma_set, 'proportional', normals, lists)))
    i_nom = np.argmin(np.sum(np.abs(np.array(grid_vector)-np.array(BIS_param_nominal))), axis=0)
    eta0[i_nom] = alpha
    return grid_vector, eta0


def load_mekf_param(point_number: list[int],
                    q: float,
                    r: float,
                    alpha: float,
                    lambda_2: float,
                    epsilon: float):
    # %% MEKF parameters
    nominal_BIS_model = pas.BIS_model()
    mean_c50p = nominal_BIS_model.c50p
    mean_c50r = nominal_BIS_model.c50r
    mean_gamma = nominal_BIS_model.gamma
    cv_c50p = 0.182
    cv_c50r = 0.888
    cv_gamma = 0.304

    BIS_param_nominal = nominal_BIS_model.hill_param
    # estimation of log normal standard deviation
    w_c50p = np.sqrt(np.log(1+cv_c50p**2))
    w_c50r = np.sqrt(np.log(1+cv_c50r**2))
    w_gamma = np.sqrt(np.log(1+cv_gamma**2))

    c50p_normal = scipy.stats.lognorm(scale=mean_c50p, s=w_c50p)
    c50r_normal = scipy.stats.lognorm(scale=mean_c50r, s=w_c50r)
    gamma_normal = scipy.stats.lognorm(scale=mean_gamma, s=w_gamma)

    nb_points = point_number[0]
    points = np.linspace(0, 1, nb_points+1)
    points = [np.mean([points[i], points[i+1]]) for i in range(nb_points)]

    c50p_list = c50p_normal.ppf(points)

    nb_points = point_number[1]
    points = np.linspace(0, 1, nb_points+1)
    points = [np.mean([points[i], points[i+1]]) for i in range(nb_points)]

    c50r_list = c50r_normal.ppf(points)

    nb_points = point_number[2]
    points = np.linspace(0, 1, nb_points+1)
    points = [np.mean([points[i], points[i+1]]) for i in range(nb_points)]
    gamma_list = gamma_normal.ppf(points)

    # surrender list by Inf value
    c50p_list = np.concatenate(([-np.Inf], c50p_list, [np.Inf]))
    c50r_list = np.concatenate(([-np.Inf], c50r_list, [np.Inf]))
    gamma_list = np.concatenate(([-np.Inf], gamma_list, [np.Inf]))

    P0 = 1e-3 * np.eye(9)
    Q = q
    Q_mat = Q * np.diag([0.1, 0.1, 0.05, 0.05, 1, 1, 10, 1, 1])
    R = r
    alpha = alpha
    list_param = [c50p_list, c50r_list, gamma_list]
    normal_list = [c50p_normal, c50r_normal, gamma_normal]
    grid_vector, eta0 = init_proba(alpha, lists=list_param, BIS_param_nominal=BIS_param_nominal, normals=normal_list)
    lambda_1 = 1
    lambda_2 = lambda_2
    nu = 1.e-5
    epsilon = epsilon

    design_param = [lambda_1, lambda_2, nu, epsilon]

    MEKF_param = {
        'Q': Q_mat,
        'R': R,
        'P0': P0,
        'grid_vector': grid_vector,
        'eta0': eta0,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
        'nu': nu,
        'epsilon': epsilon,
    }

    [Q_mat, R, P0, grid_vector, eta0, design_param]

    return MEKF_param


def load_mhe_param(vmax: float,
                   vmin: float,
                   R: float,
                   N_mhe: int,
                   q: float):
    Q = np.diag([1, 550, 550, 1, 1, 50, 750, 1]+[1e3]*4)*q
    P = np.diag([1, 550, 550, 1, 1, 50, 750, 1])

    theta = [vmin, vmax, 100, 0.02]*4
    theta[12] = vmax
    theta[13] = -vmax+vmin
    MHE_std = {
        'R': R,
        'Q': Q,
        'P': P,
        'theta': theta,
        'horizon_length': N_mhe,
    }
    return MHE_std
