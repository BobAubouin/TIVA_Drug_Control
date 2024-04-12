import numpy as np

from close_loop_anesth.mhe import MHE
from close_loop_anesth.mekf import MEKF


class MEKF_MHE:
    def __init__(self,
                 A: list,
                 B: list,
                 BIS_param: list,
                 ts: float = 1,
                 x0: list = np.zeros((8, 1)),
                 Q_mhe: list = np.eye(12),
                 P_mhe: list = np.eye(8),
                 R_mhe: list = np.array([1]),
                 horizon_length: int = 20,
                 theta: list = np.ones(16),
                 grid_vector: list = [],
                 Q_mekf: list = np.eye(9),
                 R_mekf: list = np.array([1]),
                 P0_mekf: list = np.eye(9),
                 eta0: list = None,
                 lambda_1: float = 1,
                 lambda_2: float = 1,
                 nu: float = 0.1,
                 epsilon: float = 0.9,
                 switch_time: int = 300,
                 ) -> None:

        self.switch_time = switch_time
        self.ts = ts

        A_int = np.block([[A, np.zeros((8, 1))], [np.zeros((1, 9))]])
        B_int = np.block([[B], [np.zeros((1, 2))]])

        self.mekf = MEKF(A=A_int,
                         B=B_int,
                         grid_vector=grid_vector,
                         ts=ts,
                         R=R_mekf,
                         Q=Q_mekf,
                         P0=P0_mekf,
                         eta0=eta0,
                         lambda_1=lambda_1,
                         lambda_2=lambda_2,
                         nu=nu,
                         epsilon=epsilon)

        self.mhe = MHE(A=A,
                       B=B,
                       BIS_param=BIS_param,
                       ts=ts,
                       Q=Q_mhe,
                       R=R_mhe,
                       P=P_mhe,
                       horizon_length=horizon_length,
                       theta=theta)

        self.time = 0
        self.flag_switch = False
        self.x = np.zeros((12, horizon_length))
        self.bis = np.zeros(horizon_length)
        self.u = np.zeros((2*horizon_length))

    def one_step(self, u: np.array, bis: float) -> np.array:

        self.time += self.ts
        if self.time < self.switch_time:
            x, Bis = self.mekf.one_step(u, bis)
            self.x = np.concatenate((self.x[:, 1:], x.reshape(12, 1)), axis=1)
            self.x[-3:, :] = np.repeat(self.x[-3:, -1], self.mhe.N).reshape(3, self.mhe.N)
            self.bis = np.concatenate((self.bis[1:], [bis]))
            self.u = np.concatenate((self.u[2:], u))
        else:
            if not self.flag_switch:
                self.mhe.x_pred = self.x.reshape(
                    self.mhe.nb_states*self.mhe.N, order='F')
                self.mhe.y = list(self.bis)
                self.mhe.u = self.u
                self.mhe.time = self.time
                self.flag_switch = True

            x, Bis = self.mhe.one_step(u, bis)

        return x, Bis
