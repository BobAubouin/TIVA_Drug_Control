import numpy as np


class PID():
    """Implementation of a working PID with anti-windup.

    PID = Kp ( 1 + Te / (Ti - Ti z^-1) + Td (1-z^-1) / (Td/N (1-z^-1) + Te) )
    """

    def __init__(self, Kp: float, Ti: float, Td: float, N: int = 5,
                 ts: float = 1, umax: float = 1e10, umin: float = -1e10):
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
        ts : float, optional
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
        self.ts = ts
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
        self.error = -(Bis_target - BIS)
        self.integral_part += self.ts / self.Ti * self.error

        self.derivative_part = (self.derivative_part * self.Td / self.N +
                                self.Td * (BIS - self.last_BIS)) / (self.ts +
                                                                    self.Td / self.N)
        self.last_BIS = BIS

        self.control_input = self.Kp * (self.error + self.integral_part + self.derivative_part)

        # Anti windup Conditional Integration from
        # Visioli, A. (2006). Anti-windup strategies. Practical PID control, 35-60.
        if (self.control_input >= self.umax) and self.control_input * self.error <= 0:
            self.integral_part = self.umax / self.Kp - self.error - self.derivative_part
            self.control_input = np.array(self.umax)

        elif (self.control_input <= self.umin) and self.control_input * self.error <= 0:
            self.integral_part = self.umin / self.Kp - self.error - self.derivative_part
            self.control_input = np.array(self.umin)

        return self.control_input

    def change_param(self, Kp, Ti, Td):
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td

        self.integral_part = self.control_input / self.Kp - self.error - self.derivative_part
