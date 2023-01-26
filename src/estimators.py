"""Created on Mon Apr 25 14:36:09 2022   @author: aubouinb."""

import numpy as np
import casadi as cas
from scipy.linalg import expm


def discretize(A: list, B: list, ts: float) -> (list, list):
    """Discretize LTI systems.

    Parameters
    ----------
    A : list
        Dynamic matric of the continuous system dx/dt = Ax + Bu.
    B : list
        Input matric of the continuous system dx/dt = Ax + Bu.
    ts : float, optional
        Sampling time of the system. The default is 1.

    Returns
    -------
    Ad : list
        Dynamic matric of the discret system x+ = Adx + Bdu.
    Bd : list
        Input matric of the discret system x+ = Adx + Bdu.

    """
    (n, m) = B.shape
    Mt = np.zeros((n+m, n+m))
    Mt[0:n, 0:n] = A
    Mt[0:n, n:n+m] = B
    Mtd = expm(Mt*ts/60)
    Ad = Mtd[0:n, 0:n]
    Bd = Mtd[0:n, n:n+m]
    return Ad, Bd


def BIS(xep: float, xer: float, Bis_param: list) -> float:
    """
    Compute the non-linear output function.

    Parameters
    ----------
    xep : float
        Propofol concentration in the effect site.
    xer : float
        Remifentanil concentration in the effect site.
    Bis_param : list
        Parameters of the non-linear function BIS_param = [C50p, C50r, gamma, beta, E0, Emax].

    Returns
    -------
    BIS : float
        BIS value associated to the concentrations.

    """
    C50p = Bis_param[0]
    C50r = Bis_param[1]
    gamma = Bis_param[2]
    beta = Bis_param[3]
    E0 = Bis_param[4]
    Emax = Bis_param[5]
    up = max(0, xep / C50p)
    ur = max(0, xer / C50r)
    Phi = up/(up + ur + 1e-6)
    U_50 = 1 - beta * (Phi - Phi**2)
    i = (up + ur)/U_50
    BIS = E0 - Emax * i ** gamma / (1 + i ** gamma)
    return BIS


class EKF:
    """Implementation of the Extended Kalman Filter for the Coadministration of drugs in Anesthesia."""

    def __init__(self, A: list, B: list, BIS_param: list, ts: float, x0: list = np.zeros((8, 1)),
                 Q: list = np.eye(8), R: list = np.array([1]), P0: list = np.eye(8)):
        """
        Init the EKF class.

        Parameters
        ----------
        A : list
            Dynamic matric of the continuous system dx/dt = Ax + Bu.
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
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]

        self.R = R
        self.Q = Q
        self.P = P0

        # declare CASADI variables
        x = cas.MX.sym('x', 8)  # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        y = cas.MX.sym('y')  # BIS [%]
        prop = cas.MX.sym('prop')   # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')   # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)
        P = cas.MX.sym('P', 8, 8)   # P matrix
        Pup = cas.MX.sym('P', 8, 8)   # P matrix

        # declare CASADI functions
        xpred = cas.MX(self.Ad) @ x + cas.MX(self.Bd) @ u
        Ppred = cas.MX(self.Ad) @ P @ cas.MX(self.Ad.T) + cas.MX(self.Q)
        self.Pred = cas.Function('Pred', [x, u, P], [xpred, Ppred], [
                                 'x', 'u', 'P'], ['xpred', 'Ppred'])

        up = x[3] / C50p
        ur = x[7] / C50r
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50

        h_fun = E0 - Emax * i ** gamma / (1 + i ** gamma)
        self.output = cas.Function('output', [x], [h_fun], ['x'], ['bis'])

        H = cas.gradient(h_fun, x).T

        S = H @ P @ H.T + cas.MX(self.R)
        K = P @ H.T @ cas.inv(S)

        error = y - h_fun
        xup = x + K @ error
        Pup = (cas.MX(np.identity(8)) - K@H)@P
        # S = H @ P0 @ H.T + cas.MX(self.R)
        self.Update = cas.Function('Update', [x, y, P], [xup, Pup, error, S], [
                                   'x', 'y', 'P'], ['xup', 'Pup', 'error', 'S'])

        # init state and output
        self.x = x0
        self.Biso = [BIS(self.x[3], self.x[7], BIS_param)]
        self.error = 0

    def estimate(self, u: list, bis: float) -> (list, float):
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
        self.Predk = self.Pred(x=self.x, u=u, P=self.P)
        self.xpr = self.Predk['xpred'].full().flatten()
        self.Ppr = self.Predk['Ppred'].full()

        self.Updatek = self.Update(x=self.xpr, y=bis, P=self.Ppr)
        self.x = self.Updatek['xup'].full().flatten()
        self.P = self.Updatek['Pup']
        self.error = float(self.Updatek['error'])
        self.bis_pred = bis - self.error
        self.S = self.Updatek['S']

        self.x[3] = max(1e-3, self.x[3])
        self.x[7] = max(1e-3, self.x[7])

        self.Bis = BIS(self.x[3], self.x[7], self.BIS_param)

        return self.x, self.Bis

    def predict_from_state(self, x: list, up: list, ur: list) -> list:
        """
        Return the BIS prediction using the given initial state and the control input.

        Parameters
        ----------
        x : list
            Initial state vector of the interval.
        up : list
            Propofol rates over the interval.
        ur : list
            Remifentanil rates over the interval.

        Returns
        -------
        BIS_list: list
            BIS value predicted by the model over the interval.

        """
        bis = self.output(x=x)
        BIS_list = [float(bis['bis'])]
        x = np.expand_dims(x, axis=1)
        for i in range(len(up)):
            u = np.array([[up[i]], [ur[i]]])
            x = self.Ad @ x + self.Bd @ u
            bis = self.output(x=x)
            BIS_list.append(float(bis['bis']))

        return np.array(BIS_list)
