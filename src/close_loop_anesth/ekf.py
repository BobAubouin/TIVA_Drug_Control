import numpy as np
from close_loop_anesth.utils import discretize, derivated_of_bis, compute_bis


class EKF:
    """Implementation of the Extended Kalman Filter for the Coadministration of drugs in Anesthesia."""

    def __init__(self, A: list, B: list, BIS_param: list, ts: float, x0: list = np.ones((9, 1))*1.e-3,
                 Q: list = np.eye(9), R: list = np.array([1]), P0: list = np.eye(9)):
        """
        Init the EKF class.

        Parameters
        ----------
        A : list
            Dynamic matric of the continuous system dx/dt = Ax + Bu.
            x = [x1p, x2p, x3p, xep, x1r, x2r, x3r, xer, disturbance]
        B : list
            Input matric of the continuous system dx/dt = Ax + Bu.
        BIS_param : list
            Contains parameters of the non-linear function output BIS_param = [C50p, C50r, gamma, beta, E0, Emax]
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

        Returns
        -------
        None.

        """
        self.Ad, self.Bd = discretize(A, B, ts)
        self.ts = ts
        self.BIS_param = BIS_param
        C50p, C50r, gamma, beta, E0, Emax = BIS_param

        self.R = R
        self.Q = Q
        self.P = P0

        # init state and output
        self.x = x0
        self.Biso = [compute_bis(self.x[3], self.x[7], BIS_param)]
        self.error = 0

    def Pred(self, x: np.array, u: np.array, P: np.array):
        """Compute the prediction step.

        Parameters
        ----------
        x : np.array
            State vector at time k.
        u : np.array
            Control input at time k.
        P : np.array
            Covariance matrix at time k.

        Returns
        -------
        x_pred: np.array
            Predicted state vector at time k+1.
        P_pred: np.array
            Precited covariance matrix at time k+1.
        """
        x_pred = self.Ad @ x + self.Bd @ u
        P_pred = self.Ad @ P @ self.Ad.T + self.Q
        return x_pred, P_pred

    def Update(self, x_pred: np.array, y: np.array, P: np.array):
        """
        Compute the Update sep.

        Parameters
        ----------
        x_pred : np.array
            State vector predicted at time k for k+1.
        y : np.array
            Measure at time k+1.
        P : np.array
            Covariance matrix predicted at time k for time k+1.

        Returns
        -------


        """
        H = derivated_of_bis(x_pred, self.BIS_param)
        S = H @ P @ H.T + self.R

        K = P @ H.T * 1/S

        error = y - (compute_bis(x_pred[3], x_pred[7], self.BIS_param) + self.x[8])
        xup = x_pred + K * error
        Pup = (np.identity(9) - K @ H)@P

        return xup, Pup, K, error, S

    def one_step(self, u: list, bis: float) -> tuple[list, float]:
        """
        Estimate the state given past input and current measurement.

        Parameters
        ----------
        u : list
            Last control inputs.
        bis : float
            Last BIS measurement.

        Returns
        -------
        x: list
            State estimation.
        BIS: float
            Filtered BIS.

        """
        u = np.expand_dims(u, axis=0).T

        self.xpr, self.Ppr = self.Pred(x=self.x, u=u, P=self.P)

        self.x, self.P,  self.K, self.error, self.S = self.Update(x_pred=self.xpr, y=bis, P=self.Ppr)
        self.bis_pred = bis - self.error

        self.x = np.clip(self.x, a_min=1e-3, a_max=None)
        self.bis = compute_bis(self.x[3], self.x[7], self.BIS_param) + self.x[8]

        return self.x, self.bis
