"""Created on Mon Apr 25 14:36:09 2022   @author: aubouinb."""

import numpy as np
import casadi as cas
import control as ctrl
from scipy.linalg import expm
import matplotlib.pyplot as plt


def discretize(A: list, B: list, ts: float) -> tuple[list, list]:
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
    Mtd = expm(Mt*ts)
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

    def estimate(self, u: list, bis: float) -> tuple[list, float]:
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

        # self.x[3] = max(1e-3, self.x[3])
        # self.x[7] = max(1e-3, self.x[7])
        self.x = np.clip(self.x, a_min=1e-3, a_max=None)
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

    def find_from_state(self, x: list, up: list, ur: list) -> list:
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
        invA = np.linalg.inv(self.Ad)
        bis = self.output(x=x)
        BIS_list = [float(bis['bis'])]
        x = np.expand_dims(x, axis=1)
        for i in range(len(up)):
            u = np.array([[up[len(up)-i-1]], [ur[len(up)-i-1]]])
            x = invA @ (x - self.Bd @ u)
            bis = self.output(x=x)
            BIS_list = [float(bis['bis'])] + BIS_list

        return np.array(BIS_list)


class EKF_integrator:
    """Implementation of the Extended Kalman Filter for the Coadministration of drugs in Anesthesia.

    Same than EKF but with the estimation of a constant distrubance term."""

    def __init__(self, A: list, B: list, BIS_param: list, ts: float, x0: list = np.zeros((9, 1)),
                 Q: list = np.eye(9), R: list = np.array([1]), P0: list = np.eye(9)):
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
            Initial state of the system. The default is np.zeros((9, 1)).
        Q : list, optional
            Covariance matrix of the process uncertainties. The default is np.eye(9).
        R : list, optional
            Covariance matrix of the measurement noises. The default is np.array([1]).
        P0 : list, optional
            Initial covariance matrix of the state estimation. The default is np.eye(9).
        Returns
        -------
        None.

        """
        self.Ad, self.Bd = discretize(A, B, ts)
        self.Ad = np.block([[self.Ad, np.zeros((8, 1))], [np.zeros((1, 8)), np.eye(1)]])
        self.Bd = np.block([[self.Bd], [np.zeros((1, 2))]])
        self.sys = ctrl.ss(self.Ad, self.Bd, np.eye(9), np.zeros((9, 2)), ts)
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
        x = cas.MX.sym('x', 9)  # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml] d [%]
        y = cas.MX.sym('y')  # BIS [%]
        prop = cas.MX.sym('prop')   # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')   # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)
        P = cas.MX.sym('P', 9, 9)   # P matrix
        Pup = cas.MX.sym('P', 9, 9)   # P matrix

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

        h_fun = E0 - Emax * i ** gamma / (1 + i ** gamma) + x[8]
        self.output = cas.Function('output', [x], [h_fun], ['x'], ['bis'])

        H = cas.gradient(h_fun, x).T

        S = H @ P @ H.T + cas.MX(self.R)
        K = P @ H.T @ cas.inv(S)

        error = y - h_fun
        xup = x + K @ error
        Pup = (cas.MX(np.identity(9)) - K@H)@P
        # S = H @ P0 @ H.T + cas.MX(self.R)
        self.Update = cas.Function('Update', [x, y, P], [xup, Pup, error, S], [
                                   'x', 'y', 'P'], ['xup', 'Pup', 'error', 'S'])

        # init state and output
        self.x = x0
        self.Biso = [BIS(self.x[3], self.x[7], BIS_param)]
        self.error = 0

    def estimate(self, u: list, bis: float) -> tuple[list, float]:
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

        # self.x[3] = max(1e-3, self.x[3])
        # self.x[7] = max(1e-3, self.x[7])
        self.x[:8] = np.clip(self.x[:8], a_min=1e-3, a_max=None)
        self.Bis = BIS(self.x[3], self.x[7], self.BIS_param) + self.x[8]

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
        U = np.concatenate(([up], [ur]), axis=0)
        # x = np.expand_dims(x, axis=1)
        _, yout = ctrl.forced_response(self.sys, U=U, X0=x, return_x=False)
        bis = self.output(x=x)
        BIS_list = [float(bis['bis'])]
        for i in range(len(yout[0])):
            bis = self.output(x=yout[:, i])
            BIS_list.append(float(bis['bis']))
        return np.array(BIS_list)

    def find_from_state(self, x: list, up: list, ur: list) -> list:
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
        invA = np.linalg.inv(self.Ad)
        bis = self.output(x=x)
        BIS_list = [float(bis['bis'])]
        x = np.expand_dims(x, axis=1)
        for i in range(len(up)):
            u = np.array([[up[len(up)-i-1]], [ur[len(up)-i-1]]])
            x = invA @ (x - self.Bd @ u)
            bis = self.output(x=x)
            BIS_list = [float(bis['bis'])] + BIS_list

        return np.array(BIS_list)


class MHE:
    """Implementation of the Moving Horizon Estimator for the Coadministration of propof and remifentanil in Anesthesia.

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
    N_MHE : int, optional
        Number of steps of the horizon. The default is 20.
    theta : list, optional
        Parameters of the Q matrix. The default is np.ones(12).

    Returns
    -------
    None.

    """

    def __init__(self, A: list, B: list, BIS_param: list, ts: float = 1, x0: list = np.zeros((8, 1)),
                 Q: list = np.eye(8), R: list = np.array([1]), N_MHE: int = 20, theta: list = np.ones(12)) -> None:

        self.Ad, self.Bd = discretize(A, B, ts)
        self.ts = ts
        self.nb_states = 11
        self.nb_inputs = 2
        self.BIS_param = BIS_param
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]

        self.Q = cas.MX(Q)
        self.R = R
        self.N_mhe = N_MHE

        # declare CASADI variables
        x = cas.MX.sym('x', self.nb_states)  # x1p, x2p, x3p, x4p, x1r, x2r, x3r, x4r, c50p, c50r, gamma
        u = cas.MX.sym('u', self.nb_inputs)   # Propofol and remifentanil infusion rate

        self.Ad = np.block([[self.Ad, np.zeros((8, 3))], [np.zeros((3, 8)), np.eye(3)]])
        self.Bd = np.block([[self.Bd], [np.zeros((3, 2))]])
        # declare CASADI functions
        xpred = cas.MX(self.Ad) @ x + cas.MX(self.Bd) @ u
        self.Pred = cas.Function('Pred', [x, u], [xpred], ['x', 'u'], ['xpred'])

        up = x[3] / x[8]
        ur = x[7] / x[9]
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50
        y = E0 - Emax * i ** x[10] / (1 + i ** x[10])
        self.output = cas.Function('output', [x], [y], ['x'], ['bis'])

        # ----- optimization problem -----
        # optimization variables
        x_bar = cas.MX.sym('x0', self.nb_states*N_MHE)
        # parameters
        x_pred = cas.MX.sym('x_pred', self.nb_states*N_MHE)
        u = cas.MX.sym('u', self.nb_inputs*N_MHE)
        y = cas.MX.sym('y', N_MHE)
        time = cas.MX.sym('time', 1)

        # objective function
        Q8 = theta[0] + theta[1]*np.exp(-theta[2]*np.exp(-theta[3]*time))
        Q9 = theta[4] + theta[5]*np.exp(-theta[6]*np.exp(-theta[7]*time))
        Q10 = theta[8] + theta[9]*np.exp(-theta[10]*np.exp(-theta[11]*time))
        Q = cas.blockcat([[self.Q, cas.MX(np.zeros((8, 3)))],
                          [cas.MX(np.zeros((3, 8))), cas.diag(cas.vertcat(Q8, Q9, Q10))]])

        J = (y[0] - self.output(x_bar[:self.nb_states]))**2 * self.R
        g = []
        for i in range(1, N_MHE):
            J += (y[i] - self.output(x_bar[self.nb_states*i:self.nb_states*(i+1)]))**2 * self.R
            x_pred_plus = self.Pred(x=x_pred[self.nb_states*i:self.nb_states*(i+1)],
                                    u=u[self.nb_inputs*(i-1):self.nb_inputs*i])['xpred']
            J += (x_bar[self.nb_states*i:self.nb_states*(i+1)] -
                  x_pred_plus).T @ Q @ (x_bar[self.nb_states*i:self.nb_states*(i+1)] - x_pred_plus)
            # constraints
            x_bar_plus = self.Pred(x=x_bar[self.nb_states*(i-1):self.nb_states*i],
                                   u=u[self.nb_inputs*(i-1):self.nb_inputs*i])['xpred']
            g = cas.vertcat(g, x_bar[self.nb_states*i:self.nb_states*(i+1)] - x_bar_plus)

        # create solver instance
        opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': cas.vertcat(*[u, y, x_pred, time]),
                'x': x_bar, 'g': g}  # +gbis
        self.solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        # define bound for variables
        self.lbx = ([1e-6]*8 + [0.5]*3)*N_MHE
        self.ubx = ([20]*8 + [8, 60, 5])*N_MHE
        self.lbg = [0]*self.nb_states*(N_MHE-1)
        self.ubg = [0]*self.nb_states*(N_MHE-1)

        # init state and output
        self.x = np.array([[1e-6]*8+[C50p, C50r, gamma]]).T * np.ones((1, N_MHE))
        self.y = []
        self.u = np.zeros(2*N_MHE)
        self.x_pred = self.x.reshape(11*N_MHE, order='F')
        self.time = 0

    def one_step(self, Bis, u) -> np.array:
        """solve the MHE problem for one step.

        Parameters
        ----------
        Bis : float
            BIS value at time t
        u : list
            propofol and remifentanil infusion rate at time t-1

        Returns
        -------
        np.array
            state estimation at time t

        """
        if len(self.y) == 0:
            self.y = [Bis]*self.N_mhe
        else:
            self.y = self.y[1:] + [Bis]

        self.u = np.hstack((self.u[2:], u))
        self.time += self.ts
        # solve the problem
        x0 = []
        for i in range(self.N_mhe):
            x0 += list(self.Pred(x=self.x_pred[self.nb_states*i:self.nb_states*(i+1)], u=self.u[2*i:2*(i+1)])
                       ['xpred'].full().flatten())
        res = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg,
                          p=cas.vertcat(*[self.u, self.y, self.x_pred, self.time]))
        x_bar = res['x']
        # x_bar = x0

        self.x = np.reshape(x_bar, (11, self.N_mhe), order='F')

        if self.time > 400000:
            # compute the prediction of x_pred
            self.x_pred = self.x_pred.reshape((11, self.N_mhe), order='F')
            X_pred_plus = np.zeros((11, self.N_mhe-1))
            for i in range(self.N_mhe-1):
                X_pred_plus[:, i] = self.Pred(x=self.x_pred[:, i], u=self.u[2*(i-1):2*i])['xpred'].full().flatten()
            plt.subplot(11, 1, 1)
            plt.plot(self.x[0, :], 'b', label='x estimated')
            plt.plot(self.x_pred[0, :], 'r', label='x init')
            plt.plot(X_pred_plus[0, :], 'g', label='x pred')
            plt.subplot(11, 1, 2)
            plt.plot(self.x[1, :], 'b')
            plt.plot(self.x_pred[1, :], 'r')
            plt.plot(X_pred_plus[1, :], 'g')
            plt.subplot(11, 1, 3)
            plt.plot(self.x[2, :], 'b')
            plt.plot(self.x_pred[2, :], 'r')
            plt.plot(X_pred_plus[2, :], 'g')
            plt.subplot(11, 1, 4)
            plt.plot(self.x[3, :], 'b')
            plt.plot(self.x_pred[3, :], 'r')
            plt.plot(X_pred_plus[3, :], 'g')
            plt.subplot(11, 1, 5)
            plt.plot(self.x[4, :], 'b')
            plt.plot(self.x_pred[4, :], 'r')
            plt.plot(X_pred_plus[4, :], 'g')
            plt.subplot(11, 1, 6)
            plt.plot(self.x[5, :], 'b')
            plt.plot(self.x_pred[5, :], 'r')
            plt.plot(X_pred_plus[5, :], 'g')
            plt.subplot(11, 1, 7)
            plt.plot(self.x[6, :], 'b')
            plt.plot(self.x_pred[6, :], 'r')
            plt.plot(X_pred_plus[6, :], 'g')
            plt.subplot(11, 1, 8)
            plt.plot(self.x[7, :], 'b')
            plt.plot(self.x_pred[7, :], 'r')
            plt.plot(X_pred_plus[7, :], 'g')
            plt.subplot(11, 1, 9)
            plt.plot(self.x[8, :], 'b')
            plt.plot(self.x_pred[8, :], 'r')
            plt.plot(X_pred_plus[8, :], 'g')
            plt.subplot(11, 1, 10)
            plt.plot(self.x[9, :], 'b')
            plt.plot(self.x_pred[9, :], 'r')
            plt.plot(X_pred_plus[9, :], 'g')
            plt.subplot(11, 1, 11)
            plt.plot(self.x[10, :], 'b')
            plt.plot(self.x_pred[10, :], 'r')
            plt.plot(X_pred_plus[10, :], 'g')
            plt.show()

            plt.figure()
            plt.plot(self.y, 'b', label='true bis')
            bis = [self.output(x=self.x[:, i])['bis'].full().flatten() for i in range(self.N_mhe)]
            bis_x0 = [self.output(x=self.x0[self.nb_states*i:self.nb_states*(i+1)])
                      ['bis'].full().flatten() for i in range(self.N_mhe)]
            plt.plot(bis, 'r', label='estimated bis')
            plt.legend()
            plt.show()
        self.x_pred = np.array(res['x']).reshape(self.nb_states*self.N_mhe)
        bis = float(self.output(x=self.x[:, [-1]])['bis'])
        return self.x[:, [-1]], bis


class MHE_integrator:
    """Implementation of the Moving Horizon Estimator for the Coadministration of propof and remifentanil in Anesthesia.

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
    N_MHE : int, optional
        Number of steps of the horizon. The default is 20.
    theta : list, optional
        Parameters of the Q matrix. The default is np.ones(16).

    Returns
    -------
    None.

    """

    def __init__(self, A: list, B: list, BIS_param: list, ts: float = 1, x0: list = np.zeros((8, 1)),
                 Q: list = np.eye(8), R: list = np.array([1]), N_MHE: int = 20, theta: list = np.ones(16)) -> None:

        Ad, Bd = discretize(A, B, ts)
        self.ts = ts
        self.nb_states = 12
        self.nb_inputs = 2
        self.BIS_param = BIS_param
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]

        self.Q = cas.MX(Q)
        self.R = R
        self.N_mhe = N_MHE

        # declare CASADI variables
        x = cas.MX.sym('x', self.nb_states)  # x1p, x2p, x3p, x4p, x1r, x2r, x3r, x4r, c50p, c50r, gamma
        u = cas.MX.sym('u', self.nb_inputs)   # Propofol and remifentanil infusion rate

        Ad = np.block([[Ad, np.zeros((8, 4))], [np.zeros((4, 8)), np.eye(4)]])
        Bd = np.block([[Bd], [np.zeros((4, 2))]])
        # declare CASADI functions
        xpred = cas.MX(Ad) @ x + cas.MX(Bd) @ u
        self.Pred = cas.Function('Pred', [x, u], [xpred], ['x', 'u'], ['xpred'])

        up = x[3] / x[8]
        ur = x[7] / x[9]
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50
        y = E0 - Emax * i ** x[10] / (1 + i ** x[10]) + x[11]
        self.output = cas.Function('output', [x], [y], ['x'], ['bis'])

        # ----- optimization problem -----
        # optimization variables
        x_bar = cas.MX.sym('x0', self.nb_states*N_MHE)
        # parameters
        x_pred = cas.MX.sym('x_pred', self.nb_states*N_MHE)
        u = cas.MX.sym('u', self.nb_inputs*N_MHE)
        y = cas.MX.sym('y', N_MHE)
        time = cas.MX.sym('time', 1)

        # objective function
        Q8 = theta[0] + theta[1]*np.exp(-theta[2]*np.exp(-theta[3]*time))
        Q9 = theta[4] + theta[5]*np.exp(-theta[6]*np.exp(-theta[7]*time))
        Q10 = theta[8] + theta[9]*np.exp(-theta[10]*np.exp(-theta[11]*time))
        Q11 = theta[12] + theta[13]*(1-np.exp(-theta[14]*np.exp(-theta[15]*time)))
        Q = cas.blockcat([[self.Q, cas.MX(np.zeros((8, 4)))],
                          [cas.MX(np.zeros((4, 8))), cas.diag(cas.vertcat(Q8, Q9, Q10, Q11))]])

        J = (y[0] - self.output(x_bar[:self.nb_states]))**2 * self.R
        g = []
        for i in range(1, N_MHE):
            J += (y[i] - self.output(x_bar[self.nb_states*i:self.nb_states*(i+1)]))**2 * self.R
            x_pred_plus = self.Pred(x=x_pred[self.nb_states*i:self.nb_states*(i+1)],
                                    u=u[self.nb_inputs*(i-1):self.nb_inputs*i])['xpred']
            J += (x_bar[self.nb_states*i:self.nb_states*(i+1)] -
                  x_pred_plus).T @ Q @ (x_bar[self.nb_states*i:self.nb_states*(i+1)] - x_pred_plus)
            # constraints
            x_bar_plus = self.Pred(x=x_bar[self.nb_states*(i-1):self.nb_states*i],
                                   u=u[self.nb_inputs*(i-1):self.nb_inputs*i])['xpred']
            g = cas.vertcat(g, x_bar[self.nb_states*i:self.nb_states*(i+1)] - x_bar_plus)

        # create solver instance
        opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': cas.vertcat(*[u, y, x_pred, time]),
                'x': x_bar, 'g': g}  # +gbis
        self.solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        # define bound for variables
        self.lbx = ([1e-6]*8 + [0.5]*3 + [-20])*N_MHE
        self.ubx = ([20]*8 + [8, 60, 5, 20])*N_MHE
        self.lbg = [0]*self.nb_states*(N_MHE-1)
        self.ubg = [0]*self.nb_states*(N_MHE-1)

        # init state and output
        self.x = np.array([[1e-6]*8+[C50p, C50r, gamma, 0]]).T * np.ones((1, N_MHE))
        self.y = []
        self.u = np.zeros(2*N_MHE)
        self.x_pred = self.x.reshape(self.nb_states*N_MHE, order='F')
        self.time = 0

    def one_step(self, Bis, u) -> np.array:
        """solve the MHE problem for one step.

        Parameters
        ----------
        Bis : float
            BIS value at time t
        u : list
            propofol and remifentanil infusion rate at time t-1

        Returns
        -------
        np.array
            state estimation at time t

        """
        if len(self.y) == 0:
            self.y = [Bis]*self.N_mhe
        else:
            self.y = self.y[1:] + [Bis]

        self.u = np.hstack((self.u[2:], u))
        self.time += self.ts
        # solve the problem
        x0 = []
        for i in range(self.N_mhe):
            if len(self.x_pred[self.nb_states*i:self.nb_states*(i+1)]) != 12:
                print(0)
            x0 += list(self.Pred(x=self.x_pred[self.nb_states*i:self.nb_states*(i+1)], u=self.u[2*i:2*(i+1)])
                       ['xpred'].full().flatten())
        res = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg,
                          p=cas.vertcat(*[self.u, self.y, self.x_pred, self.time]))
        x_bar = res['x']
        # x_bar = x0

        self.x = np.reshape(x_bar, (self.nb_states, self.N_mhe), order='F')

        if self.time > 400000:
            # compute the prediction of x_pred
            self.x_pred = self.x_pred.reshape((11, self.N_mhe), order='F')
            X_pred_plus = np.zeros((11, self.N_mhe-1))
            for i in range(self.N_mhe-1):
                X_pred_plus[:, i] = self.Pred(x=self.x_pred[:, i], u=self.u[2*(i-1):2*i])['xpred'].full().flatten()
            plt.subplot(11, 1, 1)
            plt.plot(self.x[0, :], 'b', label='x estimated')
            plt.plot(self.x_pred[0, :], 'r', label='x init')
            plt.plot(X_pred_plus[0, :], 'g', label='x pred')
            plt.subplot(11, 1, 2)
            plt.plot(self.x[1, :], 'b')
            plt.plot(self.x_pred[1, :], 'r')
            plt.plot(X_pred_plus[1, :], 'g')
            plt.subplot(11, 1, 3)
            plt.plot(self.x[2, :], 'b')
            plt.plot(self.x_pred[2, :], 'r')
            plt.plot(X_pred_plus[2, :], 'g')
            plt.subplot(11, 1, 4)
            plt.plot(self.x[3, :], 'b')
            plt.plot(self.x_pred[3, :], 'r')
            plt.plot(X_pred_plus[3, :], 'g')
            plt.subplot(11, 1, 5)
            plt.plot(self.x[4, :], 'b')
            plt.plot(self.x_pred[4, :], 'r')
            plt.plot(X_pred_plus[4, :], 'g')
            plt.subplot(11, 1, 6)
            plt.plot(self.x[5, :], 'b')
            plt.plot(self.x_pred[5, :], 'r')
            plt.plot(X_pred_plus[5, :], 'g')
            plt.subplot(11, 1, 7)
            plt.plot(self.x[6, :], 'b')
            plt.plot(self.x_pred[6, :], 'r')
            plt.plot(X_pred_plus[6, :], 'g')
            plt.subplot(11, 1, 8)
            plt.plot(self.x[7, :], 'b')
            plt.plot(self.x_pred[7, :], 'r')
            plt.plot(X_pred_plus[7, :], 'g')
            plt.subplot(11, 1, 9)
            plt.plot(self.x[8, :], 'b')
            plt.plot(self.x_pred[8, :], 'r')
            plt.plot(X_pred_plus[8, :], 'g')
            plt.subplot(11, 1, 10)
            plt.plot(self.x[9, :], 'b')
            plt.plot(self.x_pred[9, :], 'r')
            plt.plot(X_pred_plus[9, :], 'g')
            plt.subplot(11, 1, 11)
            plt.plot(self.x[10, :], 'b')
            plt.plot(self.x_pred[10, :], 'r')
            plt.plot(X_pred_plus[10, :], 'g')
            plt.show()

            plt.figure()
            plt.plot(self.y, 'b', label='true bis')
            bis = [self.output(x=self.x[:, i])['bis'].full().flatten() for i in range(self.N_mhe)]
            bis_x0 = [self.output(x=self.x0[self.nb_states*i:self.nb_states*(i+1)])
                      ['bis'].full().flatten() for i in range(self.N_mhe)]
            plt.plot(bis, 'r', label='estimated bis')
            plt.legend()
            plt.show()
        self.x_pred = np.array(res['x']).reshape(self.nb_states*self.N_mhe)
        bis = float(self.output(x=self.x[:, [-1]])['bis'])
        return self.x[:, [-1]], bis
