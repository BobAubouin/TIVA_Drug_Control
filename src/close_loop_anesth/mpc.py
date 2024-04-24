import sys
import os

import numpy as np
import casadi as cas
from scipy.linalg import solve_continuous_are
from close_loop_anesth.utils import discretize


class NMPC_integrator_multi_shooting:
    """Implementation of Non-linear MPC for the Coadministration of Propofol and Remifentanil in Anesthesia.

    state vector x = [x1p, x2p, x3p, xep, x1r, x2r, x3r, xer, c50p, c50r, gamma, disturbance]

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
        N : int, optional
            Prediction horizon of the controller. The default is 10.
        Nu : int, optional
            Control horizon of the controller. The default is 10.
        R : list, optional
            Cost matrix of the input amplitude. The default is np.diag([2, 1]).
        umax : list, optional
            maximum value for the control inputs. The default is [1e10]*2.
        umin : list, optional
            minimum value for the control inputs. The default is [0]*2.
        dumax : list, optional
            maximum value for the control inputs rates. The default is [0.2, 0.4].
        dumin : list, optional
            minimum value for the control inputs rates. The default is [-0.2, -0.4].
        bool_u_eq : bool, optional
            If True, the equilibrium input is computed at each step time. The default is False.
        bool_non_linear : bool, optional
            If True, the non-linear optimization problem is used. The default is False.
        Returns
        -------
        None.
        """

    def __init__(self,
                 A: list,
                 B: list,
                 BIS_param: list,
                 ts: float = 1,
                 N: int = 10,
                 Nu: int = 10,
                 R: list = np.diag([2, 1]),
                 umax: list = [1e10]*2,
                 umin: list = [0]*2,
                 dumax: list = [1e10, 1e10],
                 dumin: list = [-1e10, -1e10],
                 bool_u_eq: bool = True,
                 bool_non_linear: bool = False,
                 terminal_cost_factor: float = 10,) -> None:
        """Init NMPC class."""

        Ad, Bd = discretize(A, B, ts)
        self.BIS_param = BIS_param

        self.umax = umax
        self.umin = umin
        self.dumin = dumin
        self.dumax = dumax
        self.R = R  # control cost
        self.Q = np.diag([0, 0, 0, 4, 0, 0, 0, 1])  # state cost
        self.P = solve_continuous_are(A, B, self.Q, self.R)
        self.Q = cas.MX(self.Q)
        self.P = cas.MX(self.P)
        # self.Q*terminal_cost_factor  # terminal cost
        self.N = N  # horizon
        self.Nu = Nu  # control horizon
        self.bool_u_eq = bool_u_eq
        self.bool_non_linear = bool_non_linear
        if not bool_non_linear:
            self.bool_u_eq = True

        # declare CASADI variables
        self.create_casadi_functions(Ad, Bd)
        # create the optimization problem
        if bool_non_linear:
            self.create_non_linear_mpc_problem()
        else:
            self.create_linear_mpc_problem()
        # create the equilibrium input problem
        self.create_equilibrium_input_problem(Ad, Bd)

        # init the previous optimal solution
        self.U_prec = [0]*N*2
        self.X_prec = [1e-3]*N*9
        self.internal_target = None

        # init the equilibrium input
        self.ueq = [0]*2
        self.xeq_simple = [2]*4
        self.xeq = [2]*8

    def create_casadi_functions(self, Ad, Bd):
        beta = self.BIS_param[3]
        E0 = self.BIS_param[4]
        Emax = self.BIS_param[5]

        x = cas.MX.sym('x', 8)  # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        prop = cas.MX.sym('prop')  # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')  # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)
        constant_param = cas.MX.sym('constant_param', 4)  # C50p, C50r, gamma, disturbance
        xpred = cas.MX(Ad) @ x + cas.MX(Bd) @ u
        self.Pred = cas.Function('Pred', [x, u], [xpred], ['x', 'u'], ['x+'])

        up = x[3] / constant_param[0]
        ur = x[7] / constant_param[1]
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50

        bis = E0 - Emax * i ** constant_param[2] / (1 + i ** constant_param[2]) + constant_param[3]

        self.Output = cas.Function('Output', [x, constant_param], [bis], ['x', 'constant_param'], ['bis'])

    def create_non_linear_mpc_problem(self):
       # Optimization problem definition, with multiple shooting
        w = []
        self.lbw = []
        self.ubw = []
        J = 0
        # gu = []
        # gbis = []
        gx = []
        # self.lbg_u = []
        # self.ubg_u = []
        # self.lbg_bis = []
        # self.ubg_bis = []
        self.lbg_x = []
        self.ubg_x = []

        X0 = cas.MX.sym('X0', 8)
        Bis_target = cas.MX.sym('Bis_target')
        # U_prec_true = cas.MX.sym('U_prec', 2)
        constant_param = cas.MX.sym('constant_param', 4)
        R = cas.MX.sym('R', 2)
        ueq = cas.MX.sym('ueq', 2)
        X = cas.MX.sym('X', 8 * self.N)
        for k in range(self.N):
            Xk_plus_1 = X[8*k:8*k+8]  # X_(k+1)
            if k <= self.Nu-1:
                U = cas.MX.sym('U', 2)
                w += [U]
                self.lbw += self.umin
                self.ubw += self.umax
            if k == 0:
                X_k = X0
                # U_prec = U_prec_true
            else:
                X_k = X[8*(k-1):8*(k-1)+8]
                # U_prec = w[-2]

            Pred = self.Pred(x=X_k, u=U)
            Xk_plus_1_pred = Pred['x+']
            Hk = self.Output(x=Xk_plus_1, constant_param=constant_param)
            bis = Hk['bis']

            J += (bis - Bis_target)**2 + (U-ueq).T @ cas.diag(R) @ (U - ueq)
            # gu += [U-U_prec]
            # gbis += [bis]
            gx += [Xk_plus_1 - Xk_plus_1_pred]
            # self.lbg_u += self.dumin
            # self.ubg_u += self.dumax
            # self.lbg_bis += [0]
            # self.ubg_bis += [100]
            self.lbg_x += [0]*8
            self.ubg_x += [0]*8
        w += [X]
        self.lbw += [1e-3]*8*self.N
        self.ubw += [1e10]*8*self.N

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': cas.vertcat(*[X0, Bis_target, constant_param, R, ueq]),  # , U_prec_true
                'x': cas.vertcat(*w), 'g': cas.vertcat(*(gx))}  # gu + gbis +
        self.solver_mpc = cas.nlpsol('solver', 'ipopt', prob, opts)

    def create_linear_mpc_problem(self):
        # Optimization problem definition, with multiple shooting
        w = []
        self.lbw = []
        self.ubw = []
        J = 0
        # gu = []
        # gbis = []
        gx = []
        # self.lbg_u = []
        # self.ubg_u = []
        # self.lbg_bis = []
        # self.ubg_bis = []
        self.lbg_x = []
        self.ubg_x = []

        X0 = cas.MX.sym('X0', 8)
        xeq = cas.MX.sym('xeq', 8)
        # Bis_target = cas.MX.sym('Bis_target')
        # U_prec_true = cas.MX.sym('U_prec', 2)
        constant_param = cas.MX.sym('constant_param', 4)
        R = cas.MX.sym('R', 2)
        ueq = cas.MX.sym('ueq', 2)
        X = cas.MX.sym('X', 8 * self.N)
        for k in range(self.N):
            Xk_plus_1 = X[8*k:8*k+8]  # X_(k+1)
            if k <= self.Nu-1:
                U = cas.MX.sym('U', 2)
                w += [U]
                self.lbw += self.umin
                self.ubw += self.umax
            if k == 0:
                X_k = X0
                # U_prec = U_prec_true
            else:
                X_k = X[8*(k-1):8*(k-1)+8]
                # U_prec = w[-2]

            Pred = self.Pred(x=X_k, u=U)
            Xk_plus_1_pred = Pred['x+']
            # Hk = self.Output(x=Xk_plus_1, constant_param=constant_param)
            # bis = Hk['bis']

            # J += (bis - Bis_target)**2 + (U-ueq).T @ cas.diag(R) @ (U - ueq)
            J += (Xk_plus_1 - xeq).T @ self.Q @ (Xk_plus_1 - xeq) + (U - ueq).T @ cas.diag(R) @ (U - ueq)
            # gu += [U-U_prec]
            # gbis += [bis]
            gx += [Xk_plus_1 - Xk_plus_1_pred]
            # self.lbg_u += self.dumin
            # self.ubg_u += self.dumax
            # self.lbg_bis += [0]
            # self.ubg_bis += [100]
            self.lbg_x += [0]*8
            self.ubg_x += [0]*8

        J += (Xk_plus_1 - xeq).T @ self.P @ (Xk_plus_1 - xeq)
        w += [X]
        self.lbw += [1e-3]*8*self.N
        self.ubw += [1e10]*8*self.N

        opts = {'osqp.verbose': 0}  # 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': cas.vertcat(*[X0, xeq, constant_param, R, ueq]),  # , U_prec_true
                'x': cas.vertcat(*w), 'g': cas.vertcat(*(gx))}  # gu + gbis +
        self.solver_mpc = cas.qpsol('solver', 'osqp', prob, opts)

    def create_equilibrium_input_problem(self, Ad, Bd):
        simple_A = np.array([[Ad[0, 0], Ad[0, 3], 0, 0],
                             [Ad[3, 0], Ad[3, 3], 0, 0],
                             [0, 0, Ad[4, 4], Ad[4, 7]],
                             [0, 0, Ad[7, 4], Ad[7, 7]]])
        simple_B = Bd[[0, 3, 4, 7], :]
        simple_E = np.array([[Ad[0, 1], Ad[0, 2], 0, 0],
                             [Ad[3, 1], Ad[3, 2], 0, 0],
                             [0, 0, Ad[4, 5], Ad[4, 6]],
                             [0, 0, Ad[7, 5], Ad[7, 6]]])

        x_simple = cas.MX.sym('x', 4)  # x1p, xep, x1r, xer [mg/ml]
        u = cas.MX.sym('u', 2)
        x_k = cas.MX.sym('x_k', 8)
        constant_param = cas.MX.sym('constant_param', 4)
        R = cas.MX.sym('R', 2)
        internal_target = cas.MX.sym('internal_target')

        Hk = self.Output(x=cas.vertcat(*[x_simple[0],
                                         x_k[1],
                                         x_k[2],
                                         x_simple[1],
                                         x_simple[2],
                                         x_k[5],
                                         x_k[6],
                                         x_simple[3]]),
                         constant_param=constant_param)
        bis = Hk['bis']

        J = (bis - internal_target)**2

        x_plus = (cas.MX(simple_A) @ x_simple
                  + cas.MX(simple_B) @ u
                  + cas.MX(simple_E) @ x_k[1, 2, 5, 6])

        g = ((x_plus - x_simple).T @ (x_plus - x_simple)
             + (u[0]*cas.sqrt(R[0]) - u[1]*cas.sqrt(R[1]))**2)

        opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'x': cas.vertcat(u, x_simple), 'g': g, 'p': cas.vertcat(
            x_k, internal_target, constant_param, R)}
        self.solver_input = cas.nlpsol('solver', 'ipopt', prob, opts)

    def one_step(self, x: list, Bis_target: float, R: list = None) -> tuple[float, float]:
        """
        Compute the next optimal control input given the current state of the system and the BIS target.

        Parameters
        ----------
        x : list
            Last state estimation.
            x = [x1p, x2p, x3p, xep, x1r, x2r, x3r, xer, c50p, c50r, gamma, disturbance]
        Bis_target : float
            Current BIS target.
        bis_param : list, optional
            BIS model parameters. The default is None.

        Returns
        -------
        Up : float
            Propofol rates for the next sample time.
        Ur : float
            Remifentanil rates for the next sample time.

        """

        constant_param = x[8:]
        x = x[:8]

        # the init point of the optimization proble is the previous optimal solution
        w0_u = []
        w0_x = []
        for k in range(self.Nu):
            if k < self.Nu-1:
                w0_u += self.U_prec[2*(k+1):2*(k+1)+2]
            else:
                w0_u += self.U_prec[2*k:2*k+2]

        for k in range(self.N):
            if k < self.N:
                w0_x += self.X_prec[8*k:8*k+8]
            else:
                X_pred = self.Pred(x=self.X_prec[8*k:8*k+8], u=self.U_prec[2*k:2*k+2])
                w0_x += list(X_pred['x+'].full().flatten())

        w0 = w0_u + w0_x

        # Init internal target
        self.internal_target = Bis_target

        if R is not None:
            self.R = R

        if self.bool_u_eq:
            ueq = self.compute_equilibrium_input(self.R, x, constant_param)
        else:
            ueq = [0]*2
        self.ueq = ueq

        if self.bool_non_linear:
            opt_param = list(x) + [Bis_target] + list(constant_param) + list(np.diag(self.R)) + list(ueq)
        else:
            opt_param = list(x) + self.xeq + list(constant_param) + list(np.diag(self.R)) + list(ueq)

        sol = self.solver_mpc(x0=w0,
                              p=opt_param,
                              lbx=self.lbw,
                              ubx=self.ubw,
                              lbg=self.lbg_x,
                              ubg=self.ubg_x)

        w_opt = sol['x'].full().flatten()
        w_opt_u = w_opt[0:2*self.Nu]
        w_opt_x = w_opt[2*self.Nu:]

        Up = w_opt_u[::2]
        Ur = w_opt_u[1::2]
        self.U_prec = list(w_opt_u)
        self.X_prec = list(w_opt_x)

        # print for debug
        if False:
            import matplotlib.pyplot as plt
            bis = np.zeros(self.N)
            for k in range(self.N):
                Xk = w_opt_x[8*k:8*k+8]
                Hk = self.Output(x=Xk, constant_param=constant_param)
                bis[k] = float(Hk['bis'])

            fig, axs = plt.subplots(2, figsize=(6, 8))
            axs[0].plot(Up, label='propofol')
            axs[0].plot(Ur, label='remifentanil')
            axs[0].grid()
            axs[0].legend()
            axs[1].plot(bis, label='bis')
            axs[1].plot([self.internal_target]*len(bis), label='bis target')
            axs[1].grid()
            axs[1].legend()
            plt.show()
        return Up[0], Ur[0]

    def compute_equilibrium_input(self, R: list, x_k: list, constant_param: list) -> list:
        p = list(x_k) + [self.internal_target] + list(constant_param) + list(np.diag(R))
        x0 = list(self.ueq) + list(self.xeq_simple)
        # x0 = [0]*2 + [2]*4
        sol = self.solver_input(x0=x0,
                                p=(p),
                                lbx=[0]*6,
                                ubx=[20]*6,
                                lbg=[0],
                                ubg=[0])
        vect_sol = sol['x'].full().flatten()
        # print(sol['f'].full().flatten())
        ueq = vect_sol[0:2]
        # print(ueq)
        self.xeq_simple = vect_sol[2:]
        self.xeq = [self.xeq_simple[0],
                    x_k[1],
                    x_k[2],
                    self.xeq_simple[1],
                    self.xeq_simple[2],
                    x_k[5],
                    x_k[6],
                    self.xeq_simple[3]]
        # print(self.xeq - self.Pred(x=self.xeq, u=ueq)['x+'].full().flatten())
        return ueq
