import numpy as np
import copy

from close_loop_anesth.ekf import EKF


class MEKF:
    """Multi Extended Kalman Filter for estimation of the PD parameters in TIVA anesthesia.

    The considered model have the following state vector:
    x = [x1p, x2p, x3p, xep, x1r, x2r, x3r, xer, disturbance]
    but return the following state vector:
    x = [x1p, x2p, x3p, xep, x1r, x2r, x3r, xer, C50p, C50r, gamma, disturbance]

    C50p, C50r, gamma are the parameters of the non-linear function output BIS_param which are computed using the grid of EKF.

    Parameters
    ----------
    A : list
        Dynamic matrix of the continuous system dx/dt = Ax + Bu.
    B : list
        Input matrix of the continuous system dx/dt = Ax + Bu.
    grid_vector : list
        Contains a drif of parameters of the non-linear function output BIS_param = [C50p, C50r, gamma, beta, E0, Emax].
    ts : float, optional
        Sampling time of the system. The default is 1.
    x0 : list, optional
        Initial state of the system. The default is np.zeros((8, 1)).
    Q : list, optional
        Covariance matrix of the process uncertainties. The default is np.eye(8).
    R : list, optional
        Covariance matrix of the measurement noises. The default is np.array([1]).
    P0 : list, optional
        Initial covariance matrix of the state estimation. The default is np.eye(8).
    eta0 : list, optional
        Initial state of the parameter estimation. The default is np.ones((6, 1)).
    lambda_1 : float, optional
        Weight of the error in the criterion. The default is 1.
    lambda_2 : float, optional
        Weight of the error in the criterion. The default is 1.
    nu : float, optional
        Forgetting factor. The default is 0.1.
    epsilon : float, optional
        Threshold for the criterion hysteresis. The default is 0.9.
    """

    def __init__(self,
                 A: list,
                 B: list,
                 grid_vector: list,
                 ts: float = 1,
                 x0: list = np.ones((9, 1))*1.e-3,
                 Q: list = np.eye(9),
                 R: list = np.array([1]),
                 P0: list = np.eye(9),
                 eta0: list = None,
                 lambda_1: float = 1,
                 lambda_2: float = 1,
                 nu: float = 0.1,
                 epsilon: float = 0.9,) -> None:
        """Init the MEKF class."""
        self.ts = ts

        # define the set of EKF
        self.EKF_list = []
        for BIS_param in grid_vector:
            self.EKF_list.append(EKF(A, B, BIS_param, ts, x0, Q, R, P0))

        # Init the criterion
        self.grid_vector = grid_vector
        if eta0 is None:
            self.eta = np.ones((len(self.EKF_list), 1))
        self.eta = copy.deepcopy(eta0)
        self.best_index = np.argmin(eta0)

        # define the design parameters
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.nu = nu
        self.epsilon = epsilon

    def one_step(self, u: list, measurement: float) -> tuple[list, float]:
        """
        Estimate the state given past input and current measurement.

        Parameters
        ----------
        u : list
            Last control inputs.
        measurement : float
            Last BIS measurement.

        Returns
        -------
        x: list
            State estimation.
        BIS: float
            Filtered BIS.
        best_index: int
            Index of the best EKF.
        """
        # estimate the state for each EKF
        for i, ekf in enumerate(self.EKF_list):
            ekf.one_step(u, measurement)
            error = measurement - ekf.bis
            K_i = np.array(ekf.K)
            self.eta[i] += self.ts*(-self.nu*self.eta[i] + self.lambda_1 * error **
                                    2 + self.lambda_2 * (K_i.T @ K_i) * error**2)

        # compute the criterion
        possible_best_index = np.argmin(self.eta)
        if self.eta[possible_best_index] < self.epsilon * self.eta[self.best_index]:
            self.best_index = possible_best_index
            # init the criterion again
            # self.eta = np.ones(len(self.EKF_list))
            # for ekf in self.EKF_list:
            #     ekf.x = self.EKF_list[self.best_index].x

        X = self.EKF_list[self.best_index].x[:, 0]
        X = np.concatenate((X[:-1], self.grid_vector[self.best_index][:3], [X[-1]]), axis=0)

        return X, self.EKF_list[self.best_index].bis
