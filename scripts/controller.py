"""Created on Mon Apr 25 14:36:09 2022   @author: aubouinb."""

import numpy as np
import casadi as cas
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


class PID():
    """Implementation of a working PID with anti-windup.

    PID = Kp ( 1 + Te / (Ti - Ti z^-1) + Td (1-z^-1) / (Td/N (1-z^-1) + Te) )
    """

    def __init__(self, Kp: float, Ti: float, Td: float, N: int = 5,
                 Ts: float = 1, umax: float = 1e10, umin: float = -1e10):
        """
        Init the class.

        Parameters
        ----------
        Kp : float
            Gain.
        Ti : float
            Integrator time constant.
        Td : float
            Derivative time constant.
        N : int, optional
            Interger to filter the derivative part. The default is 5.
        Ts : float, optional
            Sampling time. The default is 1.
        umax : float, optional
            Upper saturation of the control input. The default is 1e10.
        umin : float, optional
            Lower saturation of the control input. The default is -1e10.

        Returns
        -------
        None.

        """
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.N = N
        self.Ts = Ts
        self.umax = umax
        self.umin = umin

        self.integral_part = 0
        self.derivative_part = 0
        self.last_BIS = 100

    def one_step(self, BIS: float, Bis_target: float) -> float:
        """Compute the next command for the PID controller.

        Parameters
        ----------
        BIS : float
            Last BIS measurement.
        Bis_target : float
            Current BIS target.

        Returns
        -------
        control_input: float
            control value computed by the PID.
        """
        error = -(Bis_target - BIS)
        self.integral_part += self.Ts / self.Ti * error

        self.derivative_part = (self.derivative_part * self.Td / self.N +
                                self.Td * (BIS - self.last_BIS)) / (self.Ts +
                                                                    self.Td / self.N)
        self.last_BIS = BIS

        control_input = self.Kp * (error + self.integral_part + self.derivative_part)

        # Anti windup Conditional Integration from
        # Visioli, A. (2006). Anti-windup strategies. Practical PID control, 35-60.
        if (control_input >= self.umax) and control_input * error <= 0:
            self.integral_part = self.umax / self.Kp - error - self.derivative_part
            control_input = np.array(self.umax)

        elif (control_input <= self.umin) and control_input * error <= 0:
            self.integral_part = self.umin / self.Kp - error - self.derivative_part
            control_input = np.array(self.umin)

        return control_input


class NMPC:
    """Implementation of Non-linear MPC for the Coadministration of Propofol and Remifentanil in Anesthesia."""

    def __init__(self, A: list, B: list, BIS_param: list, ts: float = 1,
                 N: int = 10, Nu: int = 10, R: list = np.diag([2, 1]),
                 umax: list = [1e10]*2, umin: list = [0]*2,
                 dumax: list = [0.2, 0.4], dumin: list = [-0.2, -0.4],
                 ki: float = 0):
        """
        Init NMPC class.

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
        ki : float, optional
            Integrator time constant. The default is 0.

        Returns
        -------
        None.

        """
        Ad, Bd = discretize(A, B, ts)
        self.BIS_param = BIS_param
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]

        self.umax = umax
        self.umin = umin
        self.dumin = dumin
        self.dumax = dumax
        self.R = R  # control cost
        self.N = N  # horizon
        self.Nu = Nu

        # declare CASADI variables
        x = cas.MX.sym('x', 8)  # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        prop = cas.MX.sym('prop')  # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')  # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)
        bis_param_cas = cas.MX.sym('bis_param_cas', 3)
        # declare CASADI functions
        xpred = cas.MX(Ad) @ x + cas.MX(Bd) @ u
        self.Pred = cas.Function('Pred', [x, u], [xpred], ['x', 'u'], ['x+'])

        up = x[3] / bis_param_cas[0]
        ur = x[7] / bis_param_cas[1]
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50

        bis = E0 - Emax * i ** bis_param_cas[2] / (1 + i ** bis_param_cas[2])

        self.Output = cas.Function('Output', [x, bis_param_cas], [bis], ['x', 'bis_param_cas'], ['bis'])

        self.U_prec = [0]*N*2
        # integrator
        self.ki = ki
        self.internal_target = None

        # Optimization problem definition
        w = []
        self.lbw = []
        self.ubw = []
        J = 0
        gu = []
        gbis = []
        self.lbg_u = []
        self.ubg_u = []

        X0 = cas.MX.sym('X0', 8)
        Bis_target = cas.MX.sym('Bis_target')
        U_prec_true = cas.MX.sym('U_prec', 2)
        Xk = X0
        for k in range(self.N):
            if k <= self.Nu-1:
                U = cas.MX.sym('U', 2)
                w += [U]
                self.lbw += self.umin
                self.ubw += self.umax
            if k == 0:
                Pred = self.Pred(x=X0, u=U)
                U_prec = U_prec_true
            elif k > self.Nu-1:
                Pred = self.Pred(x=Xk, u=U)
                U_prec = U
            else:
                Pred = self.Pred(x=Xk, u=U)
                U_prec = w[-2]

            Xk = Pred['x+']
            Hk = self.Output(x=Xk, bis_param_cas=bis_param_cas)
            bis = Hk['bis']

            # J+= ((bis - Bis_target)**2/100 + ((bis - Bis_target - 25)/30)**8)
            #    + (U-U_prec).T @ self.R @ (U-U_prec)

            # Ju = ((U-U_prec).T @ self.R @ (U-U_prec)/100 +
            #       (((U-U_prec).T @ self.R @ (U-U_prec)+10)/10)**4)

            # J += ((bis - Bis_target)**2/100 + ((bis - Bis_target - 30)/30)**32) + Ju

            J += (bis - Bis_target)**2 + ((U).T @ self.R @ (U))**2
            # if k == self.N-1:
            #     J += ((bis - Bis_target)**2/100 + ((bis - Bis_target - 30)/30)**32) * 1e3

            gu += [U-U_prec]
            gbis += [bis]
            self.lbg_u += dumin
            self.ubg_u += dumax

        opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': cas.vertcat(*[X0, Bis_target, U_prec_true, bis_param_cas]),
                'x': cas.vertcat(*w), 'g': cas.vertcat(*(gu))}  # +gbis
        self.solver = cas.nlpsol('solver', 'ipopt', prob, opts)

    def one_step(self, x: list, Bis_target: float, Bis_measure: float, bis_param: list = None) -> tuple[float, float]:
        """
        Compute the next optimal control input given the current state of the system and the BIS target.

        Parameters
        ----------
        x : list
            Last state estimation.
        Bis_target : float
            Current BIS target.
        Bis_measure : float
            Last BIS measure.
        bis_param : list, optional
            BIS model parameters. The default is None.

        Returns
        -------
        Up : float
            Propofol rates for the next sample time.
        Ur : float
            Remifentanil rates for the next sample time.

        """
        # the init point of the optimization proble is the previous optimal solution
        w0 = []
        for k in range(self.Nu):
            if k < self.Nu-1:
                w0 += self.U_prec[2*(k+1):2*(k+1)+2]
            else:
                w0 += self.U_prec[2*k:2*k+2]

        # Init internal target
        if self.internal_target is None:
            self.internal_target = Bis_target
        if bis_param is None:
            bis_param = self.BIS_param[:3]

        sol = self.solver(x0=w0, p=list(x) + [self.internal_target] + list(self.U_prec[0:2]) + bis_param,
                          lbx=self.lbw, ubx=self.ubw, lbg=self.lbg_u, ubg=self.ubg_u)

        w_opt = sol['x'].full().flatten()

        Up = w_opt[::2]
        Ur = w_opt[1::2]
        self.U_prec = list(w_opt)

        Hk = self.Output(x=x)
        bis = float(Hk['bis'])

        # integrator
        self.internal_target = self.internal_target + self.ki * (Bis_target - Bis_measure)

        # print for debug
        if False:
            bis = np.zeros(self.N)
            Xk = x
            for k in range(self.N):
                if k < self.Nu-1:
                    u = w_opt[2*k:2*k+2]
                else:
                    u = w_opt[-2:]
                Pred = self.Pred(x=Xk, u=u)
                Xk = Pred['x+']
                Hk = self.Output(x=Xk)
                bis[k] = float(Hk['bis'])

        if False:
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


class NMPC_integrator:
    """Implementation of Non-linear MPC for the Coadministration of Propofol and Remifentanil in Anesthesia.

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
        Returns
        -------
        None.
        """

    def __init__(self, A: list, B: list, BIS_param: list, ts: float = 1,
                 N: int = 10, Nu: int = 10, R: list = np.diag([2, 1]),
                 umax: list = [1e10]*2, umin: list = [0]*2,
                 dumax: list = [0.2, 0.4], dumin: list = [-0.2, -0.4],
                 bool_u_eq: bool = False):
        """Init NMPC class."""
        Ad, Bd = discretize(A, B, ts)
        self.BIS_param = BIS_param
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]

        self.umax = umax
        self.umin = umin
        self.dumin = dumin
        self.dumax = dumax
        self.R = R  # control cost
        self.N = N  # horizon
        self.Nu = Nu
        self.bool_u_eq = bool_u_eq

        self.simple_A = np.array([[Ad[0, 0], Ad[0, 3], 0, 0],
                                  [Ad[3, 0], Ad[3, 3], 0, 0],
                                  [0, 0, Ad[4, 4], Ad[4, 7]],
                                  [0, 0, Ad[7, 4], Ad[7, 7]]])
        self.simple_B = Bd[[0, 3, 4, 7], :]
        self.simple_E = np.array([[Ad[0, 1], Ad[0, 2], 0, 0],
                                  [Ad[3, 1], Ad[3, 2], 0, 0],
                                  [0, 0, Ad[4, 5], Ad[4, 6]],
                                  [0, 0, Ad[7, 5], Ad[7, 6]]])

        # declare CASADI variables
        x = cas.MX.sym('x', 9)  # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        prop = cas.MX.sym('prop')  # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')  # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)
        bis_param_cas = cas.MX.sym('bis_param_cas', 3)
        # declare CASADI functions

        xpred = cas.MX(Ad) @ x + cas.MX(Bd) @ u
        self.Pred = cas.Function('Pred', [x, u], [xpred], ['x', 'u'], ['x+'])

        up = x[3] / bis_param_cas[0]
        ur = x[7] / bis_param_cas[1]
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50

        bis = E0 - Emax * i ** bis_param_cas[2] / (1 + i ** bis_param_cas[2]) + x[8]

        self.Output = cas.Function('Output', [x, bis_param_cas], [bis], ['x', 'bis_param_cas'], ['bis'])

        self.U_prec = [0]*N*2
        self.internal_target = None

        # Optimization problem definition
        w = []
        self.lbw = []
        self.ubw = []
        J = 0
        gu = []
        gbis = []
        self.lbg_u = []
        self.ubg_u = []
        self.lbg_bis = []
        self.ubg_bis = []

        X0 = cas.MX.sym('X0', 9)
        Bis_target = cas.MX.sym('Bis_target')
        U_prec_true = cas.MX.sym('U_prec', 2)
        R = cas.MX.sym('R', 2)
        ueq = cas.MX.sym('ueq', 2)
        Xk = X0
        for k in range(self.N):
            if k <= self.Nu-1:
                U = cas.MX.sym('U', 2)
                w += [U]
                self.lbw += self.umin
                self.ubw += self.umax
            if k == 0:
                Pred = self.Pred(x=X0, u=U)
                U_prec = U_prec_true
            elif k > self.Nu-1:
                Pred = self.Pred(x=Xk, u=U)
                U_prec = U
            else:
                Pred = self.Pred(x=Xk, u=U)
                U_prec = w[-2]

            Xk = Pred['x+']
            Hk = self.Output(x=Xk, bis_param_cas=bis_param_cas)
            bis = Hk['bis']

            # J+= ((bis - Bis_target)**2/100 + ((bis - Bis_target - 25)/30)**8)
            #    + (U-U_prec).T @ self.R @ (U-U_prec)

            # Ju = ((U-U_prec).T @ self.R @ (U-U_prec)/100 +
            #       (((U-U_prec).T @ self.R @ (U-U_prec)+10)/10)**4)

            # J += ((bis - Bis_target)**2/100 + ((bis - Bis_target - 30)/30)**32) + Ju

            J += (bis - Bis_target)**2 + (U-ueq).T @ cas.diag(R) @ (U - ueq)
            # if k == self.N-1:
            #     J += ((bis - Bis_target)**2/100 + ((bis - Bis_target - 30)/30)**32) * 1e3

            gu += [U-U_prec]
            gbis += [bis]
            self.lbg_u += dumin
            self.ubg_u += dumax
            self.lbg_bis += [30]
            self.ubg_bis += [100]

        opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': cas.vertcat(*[X0, Bis_target, U_prec_true, bis_param_cas, R, ueq]),
                'x': cas.vertcat(*w), 'g': cas.vertcat(*(gu+gbis))}  #
        self.solver = cas.nlpsol('solver', 'ipopt', prob, opts)

    def one_step(self, x: list, Bis_target: float, R: list = None) -> tuple[float, float]:
        """
        Compute the next optimal control input given the current state of the system and the BIS target.

        Parameters
        ----------
        x : list
            Last state estimation.
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

        bis_param = x[9:12]
        x = x[:9]

        # the init point of the optimization proble is the previous optimal solution
        w0 = []
        for k in range(self.Nu):
            if k < self.Nu-1:
                w0 += self.U_prec[2*(k+1):2*(k+1)+2]
            else:
                w0 += self.U_prec[2*k:2*k+2]

        # Init internal target
        self.internal_target = Bis_target
        if bis_param is None:
            bis_param = self.BIS_param[:3]
        else:
            self.BIS_param[:3] = bis_param

        if R is not None:
            self.R = R

        if self.bool_u_eq:
            ueq = self.compute_equilibrium_input(self.R, x)
        else:
            ueq = [0]*2
        self.ueq = ueq
        sol = self.solver(x0=w0,
                          p=list(x) + [self.internal_target] + list(self.U_prec[0:2]) +
                          list(bis_param) + list(np.diag(self.R)) + list(ueq),
                          lbx=self.lbw,
                          ubx=self.ubw,
                          lbg=self.lbg_u + self.lbg_bis,
                          ubg=self.ubg_u + self.ubg_bis)

        w_opt = sol['x'].full().flatten()

        Up = w_opt[::2]
        Ur = w_opt[1::2]
        self.U_prec = list(w_opt)

        Hk = self.Output(x=x)
        bis = float(Hk['bis'])

        # print for debug
        if False:
            bis = np.zeros(self.N)
            Xk = x
            for k in range(self.N):
                if k < self.Nu-1:
                    u = w_opt[2*k:2*k+2]
                else:
                    u = w_opt[-2:]
                Pred = self.Pred(x=Xk, u=u)
                Xk = Pred['x+']
                Hk = self.Output(x=Xk)
                bis[k] = float(Hk['bis'])

        if False:
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

    def compute_equilibrium_input(self, R: list, x_k: list) -> list:

        x_simple = cas.MX.sym('x', 4)  # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        u = cas.MX.sym('u', 2)
        Hk = self.Output(x=cas.vertcat(*[x_simple[0], cas.MX(x_k[1]), cas.MX(x_k[2]), x_simple[1], x_simple[2],
                                         cas.MX(x_k[5]), cas.MX(x_k[6]), x_simple[3], cas.MX(x_k[8])]),
                         bis_param_cas=self.BIS_param[:3])
        bis = Hk['bis']
        J = (bis - self.internal_target)**2
        x_plus = self.simple_A @ x_simple + self.simple_B @ u + \
            self.simple_E @ np.array([x_k[1], x_k[2], x_k[5], x_k[6]])
        g = (x_plus - x_simple).T @ (x_plus - x_simple) + (u[0]*cas.sqrt(R[0, 0]) - u[1]*cas.sqrt(R[1, 1]))**2
        opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'x': cas.vertcat(u, x_simple), 'g': g}
        solver = cas.nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0=[0, 0] + [2] * 4, lbx=[0]*6, ubx=[20]*6, lbg=[0], ubg=[0])
        ueq = sol['x'].full().flatten()
        ueq = ueq[0:2]
        return ueq


class MMPC():
    """Implementation of a multimodel predictive control.

    From "Adaptive control using multiple models" by Narendra and Balakrishnan in 1997.
    """

    def __init__(self, estimator_list: list, controller_list: list, window_length: int = 10,
                 best_init: int = 13, hysteresis: float = 5, BIS_target: float = 50,
                 alpha: float = 0.05, beta: float = 1, lambda_p: float = 0.0001):
        """
        Init MMPC class.

        Parameters
        ----------
        estimator_list : list
            List of estimators, each one with its own BIS_model parameters.
        controller_list : list
            List of controllers, each one with its own BIS_model parameters, corresponding index with the previous list.
        window_length : int, optional
            Length of the observation window. The default is 10.
        best_init : int, optional
            Index of the initial model to use. The default is 13.
        hysteresis : float, optional
            Value of the hysteris. The default is 5.
        BIS_target : float, optional
            BIS target of the controller. The default is 50.
        alpha : float, optional
            DESCRIPTION. The default is 0.05.
        beta : float, optional
            DESCRIPTION. The default is 1.
        lambda_p : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        None.

        """
        # Save inputs in the class
        self.estimator_list = estimator_list
        self.controller_list = controller_list
        self.N_model = len(controller_list)  # total number of model
        self.window_length = window_length
        self.idx_best = best_init
        self.hysteresis = hysteresis
        self.count = 0
        self.BIS_target = BIS_target
        self.alpha = alpha
        self.beta = beta
        self.lambda_p = lambda_p
        self.ts = estimator_list[0].ts
        # Init internal variables
        self.error = np.zeros(self.N_model)
        size_state = len(estimator_list[0].x)
        self.X = np.zeros((self.N_model, self.window_length, size_state))
        self.Up = np.zeros(self.window_length-1)
        self.Ur = np.zeros(self.window_length-1)
        self.BIS = np.zeros(self.window_length)

    def one_step(self, U_prec: list, BIS: float) -> tuple[list, int]:
        """
        Compute the optimal control input using past control input value and current BIS measurement.

        First all the estimators are updated, then the best model is choosen using the minimal error prediction
        on the previous window.

        Parameters
        ----------
        U_prec : list
            Last control_input.
        BIS : float
            Last BIS measurement.

        Returns
        -------
        U: list
            Drug rates for the next sampling time.
        best_id: int
            Index of the model used to do the prediction.

        """
        plot = False  # for debug

        # init pas BIS value to the initial value
        if np.sum(self.BIS) == 0:
            self.BIS = np.ones(self.window_length)*BIS
        else:
            self.BIS = np.concatenate((self.BIS[1:], BIS), axis=0)

        for idx in range(self.N_model):
            self.estimator_list[idx].estimate(U_prec, BIS)  # update the estimators
            self.X[idx] = np.concatenate((self.X[idx, 1:, :], np.expand_dims(  # save the paste states
                self.estimator_list[idx].x, axis=1).T), axis=0)
            # compute the prediction error
            bis_pred = self.estimator_list[idx].predict_from_state(x=self.X[idx, 0], up=self.Up, ur=self.Ur)
            # bis_pred = self.estimator_list[idx].find_from_state(x=self.estimator_list[idx].x, up=self.Up, ur=self.Ur)

            epsilon = np.square(bis_pred - self.BIS)
            integral = self.ts * np.sum([np.exp(- self.lambda_p * (self.window_length - i)*self.ts) * epsilon[i]
                                         for i in range(self.window_length)])

            self.error[idx] = self.alpha*epsilon[-1] + self.beta*integral

            if plot:
                plt.plot(bis_pred, label='model ' + str(idx))

        if plot:
            # plt.legend()
            plt.plot(self.BIS, 'r--', label='Measured BIS')
            plt.show()
            plt.bar(np.arange(len(self.error)), self.error)
            plt.yscale('log')
            plt.show()

        error_min = min(self.error)
        idx_best_list = [i for i, j in enumerate(self.error) if j == error_min]
        idx_best_new = idx_best_list[0]
        self.count = min(self.window_length, self.count + 1)

        if abs(self.error[idx_best_new] - self.error[self.idx_best]) > self.hysteresis * self.count/self.window_length:
            self.idx_best = idx_best_new

        self.controller_list[self.idx_best].U_prec[0:2] = U_prec
        uP, uR = self.controller_list[self.idx_best].one_step(
            self.estimator_list[self.idx_best].x, self.BIS_target, self.estimator_list[self.idx_best].Bis)

        # save past inputs
        self.Up = np.concatenate((self.Up[1:], np.array([uP])), axis=0)
        self.Ur = np.concatenate((self.Ur[1:], np.array([uR])), axis=0)

        return [uP, uR], self.idx_best


class MMPC_integrator():
    """Implementation of a multimodel predictive control.

    From "Adaptive control using multiple models" by Narendra and Balakrishnan in 1997.
    """

    def __init__(self, estimator_list: list, controller: list, window_length: int = 10,
                 best_init: int = 13, hysteresis: float = 5, BIS_target: float = 50,
                 alpha: float = 0.05, beta: float = 1, lambda_p: float = 0.0001):
        """
        Init MMPC class.

        Parameters
        ----------
        estimator_list : list
            List of estimators, each one with its own BIS_model parameters.
        controller : list
            NMPC_integrator instance.
        window_length : int, optional
            Length of the observation window. The default is 10.
        best_init : int, optional
            Index of the initial model to use. The default is 13.
        hysteresis : float, optional
            Value of the hysteris. The default is 5.
        BIS_target : float, optional
            BIS target of the controller. The default is 50.
        alpha : float, optional
            DESCRIPTION. The default is 0.05.
        beta : float, optional
            DESCRIPTION. The default is 1.
        lambda_p : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        None.

        """
        # Save inputs in the class
        self.estimator_list = estimator_list
        self.controller = controller
        self.N_model = len(estimator_list)  # total number of model
        self.window_length = window_length
        self.idx_best = best_init
        self.hysteresis = hysteresis
        self.count = 0
        self.BIS_target = BIS_target
        self.alpha = alpha
        self.beta = beta
        self.lambda_p = lambda_p
        self.ts = estimator_list[0].ts
        # Init internal variables
        self.error = np.zeros(self.N_model)
        size_state = len(estimator_list[0].x)
        self.X = np.zeros((self.N_model, self.window_length, size_state))
        self.Up = np.zeros(self.window_length-1)
        self.Ur = np.zeros(self.window_length-1)
        self.BIS = np.zeros(self.window_length)

    def one_step(self, U_prec: list, BIS: float, R: list = None) -> tuple[list, int]:
        """
        Compute the optimal control input using past control input value and current BIS measurement.

        First all the estimators are updated, then the best model is choosen using the minimal error prediction
        on the previous window.

        Parameters
        ----------
        U_prec : list
            Last control_input.
        BIS : float
            Last BIS measurement.

        Returns
        -------
        U: list
            Drug rates for the next sampling time.
        best_id: int
            Index of the model used to do the prediction.

        """
        plot = False  # for debug

        # init pas BIS value to the initial value
        if np.sum(self.BIS) == 0:
            self.BIS = np.ones(self.window_length)*BIS
        else:
            self.BIS = np.concatenate((self.BIS[1:], BIS), axis=0)

        for idx in range(self.N_model):
            self.estimator_list[idx].estimate(U_prec, BIS)  # update the estimators
            self.X[idx] = np.concatenate((self.X[idx, 1:, :], np.expand_dims(  # save the paste states
                self.estimator_list[idx].x, axis=1).T), axis=0)
            # compute the prediction error
            bis_pred = self.estimator_list[idx].predict_from_state(x=self.X[idx, 0], up=self.Up, ur=self.Ur)
            # bis_pred = self.estimator_list[idx].find_from_state(x=self.estimator_list[idx].x, up=self.Up, ur=self.Ur)

            epsilon = np.square(bis_pred - self.BIS)
            integral = self.ts * np.sum([np.exp(- self.lambda_p * (self.window_length - i)*self.ts) * epsilon[i]
                                         for i in range(self.window_length)])

            self.error[idx] = self.alpha*epsilon[-1] + self.beta*integral

            if plot:
                plt.plot(bis_pred, label='model ' + str(idx))

        if plot:
            # plt.legend()
            plt.plot(self.BIS, 'r--', label='Measured BIS')
            plt.show()
            plt.bar(np.arange(len(self.error)), self.error)
            plt.yscale('log')
            plt.show()

        error_min = min(self.error)
        idx_best_list = [i for i, j in enumerate(self.error) if j == error_min]
        idx_best_new = idx_best_list[0]
        self.count = min(self.window_length, self.count + 1)

        if abs(self.error[idx_best_new] - self.error[self.idx_best]) > self.hysteresis * self.count/self.window_length:
            self.idx_best = idx_best_new

        best_Bis_parameters = self.estimator_list[self.idx_best].BIS_param[:3]
        uP, uR = self.controller.one_step(
            self.estimator_list[self.idx_best].x,
            self.BIS_target,
            best_Bis_parameters,
            R)

        # save past inputs
        self.Up = np.concatenate((self.Up[1:], np.array([uP])), axis=0)
        self.Ur = np.concatenate((self.Ur[1:], np.array([uR])), axis=0)

        return [uP, uR], self.idx_best
