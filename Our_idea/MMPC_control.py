"""
Created on Wed Jan  4 14:30:58 2023

@author: aubouinb
"""

import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
from TIVA_Drug_Control.src.estimators import discretize


class Bayes_MMPC_best():
    """Implementation of Multiples models Predictive Control with Baye formula for model choice."""

    def __init__(self, estimator_list: list, controller_list: list, K: list = None,
                 Pinit: list = None, hysteresis: float = 0.05, BIS_target: float = 50):
        """Init the class Bayes MMPC.

        Inputs: - estimator_list: list of estimators, each one with its own BIS_model parameters
                - controller_list: list of controllers, each one with its own BIS_model parameters,
                                   corresponding index with the previous list
                - K: gain matrix for baye formula
                - Pinit: prior probability of each model
                - hysteresis: value of the hysteris (as a probability) to switch between models
        """
        self.estimator_list = estimator_list
        self.controller_list = controller_list
        self.N_model = len(controller_list)  # total number of model
        if K is not None:
            self.K = K
        else:
            self.K = 1
        if Pinit is not None:
            self.Pinit = Pinit
        else:
            self.Pinit = np.ones(self.N_model)/self.N_model

        self.P = self.Pinit
        self.hysteresis = hysteresis
        self.BIS_target = BIS_target
        # init best model
        self.idx_best = 13

    def one_step(self, U_prec, BIS):
        """Compute the optimal control input using past control input value and current BIS measurement.

        First all the estimator are updated and the model probabilty are updated using the baye formula.
        Then the best model is choosed as the one with the higher probability with a small hysteresis to
        avoid instability. This model is then used in a Non linear MPC to obtain the next control input.
        """
        # update the estimators
        self.S = []
        self.invS = []
        self.nS = []
        self.bis_pred = []
        for idx in range(self.N_model):
            self.estimator_list[idx].estimate(U_prec, BIS)
            self.S.append(self.estimator_list[idx].S)
            self.nS.append(abs(self.S[-1]))
            self.bis_pred.append(self.estimator_list[idx].bis_pred)

        if False:
            plt.bar(np.arange(len(self.bis_pred))+1, self.bis_pred)
            plt.bar(0, BIS)
            plt.show()
        # update the probaility

        Ptot = np.sum([1/np.sqrt(self.nS[idx]) * np.exp(-0.5*self.estimator_list[idx].error**2 / self.nS[idx])
                       * self.P[idx] for idx in range(self.N_model)])

        for idx in range(self.N_model):
            self.P[idx] = (1/np.sqrt(self.nS[idx])
                           * np.exp(-0.5*self.estimator_list[idx].error**2 / self.nS[idx]) * self.P[idx])/Ptot

        # Ptot = np.sum([np.exp(-0.5*self.estimator_list[idx].error * self.K * self.estimator_list[idx].error)
        #                * self.P[idx] for idx in range(self.N_model)])

        # for idx in range(self.N_model):
        #     self.P[idx] = (np.exp(-0.5*self.estimator_list[idx].error*self.K*self.estimator_list[idx].error)
        #                    * self.P[idx])/Ptot

        self.P = np.clip(self.P, a_min=1/(5*self.N_model), a_max=None)
        self.P = self.P/np.sum(self.P)

        proba_max = max(self.P)
        idx_best_list = [i for i, j in enumerate(self.P) if j == proba_max]
        idx_best_new = idx_best_list[0]
        if abs(self.P[idx_best_new] - self.P[self.idx_best]) > self.hysteresis:
            self.idx_best = idx_best_new

        self.controller_list[self.idx_best].U_prec[0:2] = U_prec
        uP, uR = self.controller_list[self.idx_best].one_step(
            self.estimator_list[self.idx_best].x, self.BIS_target, self.estimator_list[self.idx_best].Bis)
        if False:
            plt.bar(np.arange(len(self.P)), self.P)
            plt.show()
        if False:
            error = [abs(self.estimator_list[idx].error) for idx in range(self.N_model)]
            plt.bar(np.arange(len(error)), error)
            plt.bar(np.arange(len(error)), np.array(self.nS)/10000, alpha=0.5)
            plt.show()
            plt.pause(0.2)
        return [uP, uR], self.idx_best  # self.S[27]


class Bayes_MMPC_mean():
    """Implementation of Multiples models Predictive Control with Baye formula for model choice."""

    def __init__(self, estimator_list: list, controller: list, Bis_param_list: list,
                 K: list = None, Pinit: list = None, BIS_target: float = 50):
        """Init the class Baye MMPC.

        Inputs: - estimator_list: list of estimators, each one with its own BIS_model parameters
                - Bis_param_list: list of the BIS_parameters associated with the BIS_model,
                                   corresponding index with the previous list
                - controller: NMPC controller able to update his model
                - K: gain matrix for baye formula
                - Pinit: prior probability of each model
                - hysteresis: value of the hysteris (as a probability) to switch between models
        """
        self.estimator_list = estimator_list
        self.controller = controller
        self.Bis_param_list = Bis_param_list
        self.N_model = len(estimator_list)  # total number of model
        if K is not None:
            self.K = K
        else:
            self.K = 1
        if Pinit is not None:
            self.Pinit = Pinit
        else:
            self.Pinit = np.ones(self.N_model)/self.N_model

        self.P = self.Pinit
        self.BIS_target = BIS_target
        # init best model
        self.idx_best = 0
        # init weights
        self.W = np.ones(self.N_model)/self.N_model

    def one_step(self, U_prec, BIS):
        """Compute the optimal control input using past control input value and current BIS measurement.

        First all the estimator are updated and the model probabilty are updated using the baye formula.
        Then the best model is choosed as the one with the higher probability with a small hysteresis to
        avoid instability. This model is then used in a linear MPC to obtain the next control input.
        """
        # update the estimators
        self.S = []
        self.invS = []
        self.detS = []
        for idx in range(self.N_model):
            self.estimator_list[idx].estimate(U_prec, BIS)
            self.S.append(self.estimator_list[idx].S)
            self.invS.append(np.linalg.inv(self.S[-1]))
            self.detS.append(np.linalg.det(self.S[-1]))
        # update the probaility

        # Ptot = np.sum([1/np.sqrt(self.detS[idx]) * np.exp(-0.5*self.estimator_list[idx].error *
        #                                                   self.invS[idx] * self.estimator_list[idx].error)
        #                * self.P[idx] for idx in range(self.N_model)])

        # for idx in range(self.N_model):
        #     self.P[idx] = (1/np.sqrt(self.detS[idx])
        #                    * np.exp(-0.5*self.estimator_list[idx].error*self.invS[idx]*self.estimator_list[idx].error)
        #                    * self.P[idx])/Ptot
        Ptot = np.sum([np.exp(-0.5*self.estimator_list[idx].error * self.K * self.estimator_list[idx].error)
                       * self.P[idx] for idx in range(self.N_model)])

        for idx in range(self.N_model):
            self.P[idx] = (np.exp(-0.5*self.estimator_list[idx].error*self.K*self.estimator_list[idx].error)
                           * self.P[idx])/Ptot

        self.P = np.clip(self.P, a_min=1/(5*self.N_model), a_max=None)

        for idx in range(self.N_model):
            self.W[idx] = self.P[idx] / np.sum(self.P)

        self.X = np.sum([self.W[idx] * self.estimator_list[self.idx_best].x for idx in range(self.N_model)], axis=0)
        self.BIS = np.sum([self.W[idx] * self.estimator_list[self.idx_best].Bis for idx in range(self.N_model)], axis=0)
        self.Bis_param = np.sum([self.W[idx] * np.array(self.Bis_param_list[idx])
                                for idx in range(self.N_model)], axis=0)

        self.controller.update_model(self.Bis_param)
        uP, uR = self.controller.one_step(self.X, self.BIS_target, self.BIS)
        return [uP, uR], self.W[27]


class Euristic_MMPC():
    def __init__(self, estimator_list: list, controller_list: list, window_length: int = 10,
                 best_init: int = 13, hysteresis: float = 5, BIS_target: float = 50):
        """Init the class Euristic_MMPC.

        Inputs: - estimator_list: list of estimators, each one with its own BIS_model parameters
                - controller_list: list of controllers, each one with its own BIS_model parameters,
                                   corresponding index with the previous list
                - window_length: length of the observation window
                - best_init: index of the initial model to use
                - hysteresis: value of the hysteris
                - BIS_target: BIS target of the controller
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

        # Init internal variables
        self.error = np.zeros(self.N_model)
        self.X = np.zeros((self.N_model, self.window_length, 8))
        self.Up = np.zeros(self.window_length-1)
        self.Ur = np.zeros(self.window_length-1)
        self.BIS = np.zeros(self.window_length)

    def one_step(self, U_prec, BIS):
        """Compute the optimal control input using past control input value and current BIS measurement.

        First all the estimators are updated, then the best model is choosen using the minimal error prediction
        on the previous window
        """
        plot = False  # for debug

        # init pas BIS value to the initial value
        if np.sum(self.BIS) == 0:
            self.BIS = np.ones(self.window_length)*BIS
        else:
            self.BIS = np.concatenate((self.BIS[1:], np.array([BIS])), axis=0)

        for idx in range(self.N_model):
            self.estimator_list[idx].estimate(U_prec, BIS)  # update the estimators
            self.X[idx] = np.concatenate((self.X[idx, 1:, :], np.expand_dims(  # save the paste states
                self.estimator_list[idx].x, axis=1).T), axis=0)
            # compute the prediction error
            bis_pred = self.estimator_list[idx].predict_from_state(x=self.X[idx, 0], up=self.Up, ur=self.Ur)
            self.error[idx] = np.sum(np.square(bis_pred - self.BIS))
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


class Euristic_Mirko():
    def __init__(self, estimator_list: list, controller_list: list, BIS_param_list: list, A: list, B: list,
                 window_length: int = 10, best_init: int = 13, hysteresis: float = 5, BIS_target: float = 50):
        """Init the class Euristic_MMPC.

        Inputs: - estimator_list: list of estimators, each one with its own BIS_model parameters
                - controller_list: list of controllers, each one with its own BIS_model parameters,
                                   corresponding index with the previous list
                - window_length: length of the observation window
                - best_init: index of the initial model to use
                - hysteresis: value of the hysteris
                - BIS_target: BIS target of the controller
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

        # Init internal variables
        self.error = np.zeros(self.N_model)
        self.X = np.zeros((self.N_model, self.window_length, 8))
        self.Up = np.zeros(self.window_length-1)
        self.Ur = np.zeros(self.window_length-1)
        self.BIS = np.zeros(self.window_length)

        # define optimization class
        self.optimization_prob = []
        for idx in range(self.N_model):
            self.optimization_prob.append(solve(BIS_param_list[idx], A, B, self.window_length))

    def one_step(self, U_prec, BIS):
        """Compute the optimal control input using past control input value and current BIS measurement.

        First all the estimators are updated, then the best model is choosen using the minimal error prediction
        on the previous window
        """
        plot = False  # for debug

        # init pas BIS value to the initial value
        if np.sum(self.BIS) == 0:
            self.BIS = np.ones(self.window_length)*BIS
        else:
            self.BIS = np.concatenate((self.BIS[1:], np.array([BIS])), axis=0)

        for idx in range(self.N_model):
            self.estimator_list[idx].estimate(U_prec, BIS)  # update the estimators
            self.X[idx] = np.concatenate((self.X[idx, 1:, :], np.expand_dims(  # save the paste states
                self.estimator_list[idx].x, axis=1).T), axis=0)
            # compute the prediction error
            self.error[idx], bis_pred = self.optimization_prob[idx].one_step(self.BIS, self.Up, self.Ur)
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


class solve():
    def __init__(self, BIS_param: list, A: list, B: list, N: int):

        self.Ad, self.Bd = discretize(A, B, 2)
        self.BIS_param = BIS_param
        C50p = BIS_param[0]
        C50r = BIS_param[1]
        gamma = BIS_param[2]
        beta = BIS_param[3]
        E0 = BIS_param[4]
        Emax = BIS_param[5]
        self.N = N

        # define systems model

        # declare CASADI variables
        x = cas.MX.sym('x', 8)  # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        prop = cas.MX.sym('prop')   # Propofol infusion rate [mg/ml/min]
        rem = cas.MX.sym('rem')   # Remifentanil infusion rate [mg/ml/min]
        u = cas.vertcat(prop, rem)

        # declare CASADI functions
        xpred = cas.MX(self.Ad) @ x + cas.MX(self.Bd) @ u
        self.Pred = cas.Function('Pred', [x, u], [xpred], ['x', 'u'], ['x+'])

        up = x[3] / C50p
        ur = x[7] / C50r
        Phi = up/(up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur)/U_50

        bis = E0 - Emax * i ** gamma / (1 + i ** gamma)

        self.Output = cas.Function('Output', [x], [bis], ['x'], ['bis'])

        # optimization problem
        # Start with an empty NLP
        w = []
        self.lbw = []
        self.ubw = []
        J = 0
        g = []
        self.lbg = []
        self.ubg = []
        BIS = cas.MX.sym('BIS', N)
        Up = cas.MX.sym('Up', N-1)
        Ur = cas.MX.sym('Ur', N-1)
        X0 = cas.MX.sym('X0', 8)

        Xk = X0

        for k in range(self.N):
            w += [Xk]
            self.lbw += [0]*8
            self.ubw += [1e3]*8
            Hk = self.Output(x=Xk)
            y = Hk['bis']
            J += (y - BIS[k])**2

            if k < self.N-1:
                Pred = self.Pred(x=Xk, u=cas.vertcat(Up[k], Ur[k]))
                Xk_1 = Pred['x+']
                Xk = cas.MX.sym('Xk_'+str(k+1), 8)

                g += [Xk_1-Xk]
                self.lbg += [-1e-8]*8
                self.ubg += [1e-8]*8

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': cas.vertcat(*[BIS, Up, Ur]), 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
        self.solver = cas.nlpsol('solver', 'ipopt', prob, opts)

        # starting point
        self.w0 = [0]*8*self.N

    def one_step(self, bis, up, ur):
        """Do the optimization process."""
        sol = self.solver(x0=self.w0, p=list(bis) + list(up) + list(ur),
                          lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)

        self.w0 = sol['x'].full().flatten()
        self.J = float(sol['f'])

        return self.J, self.w0
