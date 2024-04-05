import casadi as cas
import numpy as np
import matplotlib.pyplot as plt

from close_loop_anesth.utils import discretize


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
    horizon_length : int, optional
        Number of steps of the horizon. The default is 20.
    theta : list, optional
        Parameters of the Q matrix. The default is np.ones(16).

    Returns
    -------
    None.

    """

    def __init__(self, A: list,
                 B: list,
                 BIS_param: list,
                 ts: float = 1,
                 x0: list = np.zeros((8, 1)),
                 Q: list = np.eye(12),
                 P: list = np.eye(8),
                 R: list = np.array([1]),
                 horizon_length: int = 20,
                 theta: list = np.ones(16),
                 ) -> None:

        Ad, Bd = discretize(A, B, ts)
        self.Ad = np.block([[Ad, np.zeros((8, 4))], [np.zeros((4, 8)), np.eye(4)]])
        self.Bd = np.block([[Bd], [np.zeros((4, 2))]])

        self.ts = ts
        self.nb_states = 12
        self.nb_inputs = 2
        self.BIS_param = BIS_param
        self.Q = cas.MX(Q)
        self.P = P
        self.R = R
        self.N = horizon_length
        self.theta = theta

        C50p, C50r, gamma, _, _, _ = BIS_param

        # system dynamics
        self.declare_dynamic_functions()

        # ----- optimization problem -----
        self.define_opt_problem()

        # define bound for variables
        self.lbx = ([1e-6]*8 + [0.5]*3 + [-20])*horizon_length
        self.ubx = ([20]*8 + [8, 60, 5, 20])*horizon_length

        # init state and output
        self.x = np.array([[1e-6]*8+[C50p, C50r, gamma, 0]]).T * np.ones((1, horizon_length))
        self.y = []
        self.u = np.zeros(2*horizon_length)
        self.x_pred = self.x.reshape(self.nb_states*horizon_length, order='F')
        self.time = 0

    def declare_dynamic_functions(self):
        _, _, _, beta, E0, Emax = self.BIS_param

        # declare CASADI variables
        x = cas.MX.sym('x', self.nb_states)  # x1p, x2p, x3p, x4p, x1r, x2r, x3r, x4r, c50p, c50r, gamma, disturbance
        u = cas.MX.sym('u', self.nb_inputs)   # Propofol and remifentanil infusion rate

        # declare CASADI functions
        xpred = cas.MX(self.Ad) @ x + cas.MX(self.Bd) @ u
        self.Pred = cas.Function('Pred', [x, u], [xpred], ['x', 'u'], ['xpred'])

        up = x[3] / x[8]
        ur = x[7] / x[9]
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50
        y = E0 - Emax * i ** x[10] / (1 + i ** x[10]) + x[11]
        self.output = cas.Function('output', [x], [y], ['x'], ['bis'])

    def theta_dependancy(self, theta, time):
        return theta[0] + theta[1]*np.exp(-theta[2]*np.exp(-theta[3]*time))

    def define_opt_problem(self):
        # ----- optimization problem -----
        # optimization variables
        x_bar = cas.MX.sym('x0', self.nb_states*self.N)
        # parameters
        x_pred0 = cas.MX.sym('x_pred', self.nb_states)
        u = cas.MX.sym('u', self.nb_inputs*self.N)
        y = cas.MX.sym('y', self.N)
        time = cas.MX.sym('time', 1)

        # objective function
        J = 0
        P8 = self.theta_dependancy(self.theta[0:4], time)
        P9 = self.theta_dependancy(self.theta[4:8], time)
        P10 = self.theta_dependancy(self.theta[8:12], time)
        P11 = self.theta_dependancy(self.theta[12:16], time)

        P = cas.blockcat([[self.P, cas.MX(np.zeros((8, 4)))],
                          [cas.MX(np.zeros((4, 8))), cas.diag(cas.vertcat(P8, P9, P10, P11))]])

        for i in range(0, self.N):
            x_i = x_bar[self.nb_states*i:self.nb_states*(i+1)]
            u_i = u[self.nb_inputs*(i):self.nb_inputs*(i+1)]
            # cost function
            J += (y[i] - self.output(x_i))**2 * self.R
            if i < self.N-1:
                x_i_plus = x_bar[self.nb_states*(i+1):self.nb_states*(i+2)]
                x_bar_plus = self.Pred(x=x_i, u=u_i)['xpred']
                J += (x_i_plus - x_bar_plus).T @ self.Q @ (x_i_plus - x_bar_plus)
            if i == 0:
                J += (x_i - x_pred0).T @ P @ (x_i - x_pred0)

        # create solver instance
        opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': cas.vertcat(*[u, y, x_pred0, time]),
                'x': x_bar}  # +gbis
        self.solver = cas.nlpsol('solver', 'ipopt', prob, opts)

    def one_step(self, u, Bis) -> np.array:
        """solve the MHE problem for one step.

        Parameters
        ----------
        u : list
            propofol and remifentanil infusion rate at time t-1
        Bis : float
            BIS value at time t

        Returns
        -------
        np.array
            state estimation at time t

        """
        if len(self.y) == 0:
            self.y = [Bis]*self.N
        else:
            self.y = self.y[1:] + [Bis]

        self.u = np.hstack((self.u[2:], u))
        self.time += self.ts
        # init the problem
        x0 = []
        for i in range(self.N):
            if len(self.x_pred[self.nb_states*i:self.nb_states*(i+1)]) != 12:
                print(0)
            x0 += list(self.Pred(x=self.x_pred[self.nb_states*i:self.nb_states*(i+1)], u=self.u[2*i:2*(i+1)])
                       ['xpred'].full().flatten())
        x_pred_0 = self.x_pred[self.nb_states: 2*self.nb_states]
        # solve the problem
        res = self.solver(x0=x0,
                          lbx=self.lbx,
                          ubx=self.ubx,
                          p=cas.vertcat(*[self.u, self.y, x_pred_0, self.time]))
        x_bar = res['x']
        # x_bar = x0

        self.x = np.reshape(x_bar, (self.nb_states, self.N), order='F')

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
        self.x_pred = np.array(res['x']).reshape(self.nb_states*self.N)
        bis = float(self.output(x=self.x[:, [-1]])['bis'])
        return self.x[:, -1], bis
