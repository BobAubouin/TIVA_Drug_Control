import numpy as np
from scipy.linalg import expm


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


def derivated_of_bis(x: list, bis_param: list) -> list:
    """Compute the derivated of the non-linear function BIS.

    Parameters
    ----------
    x : list
        State vector.
    bis_param : list
        Parameters of the non-linear function BIS_param = [C50p, C50r, gamma, beta, E0, Emax].

    Returns
    -------
    list
        Derivated of the non-linear function BIS.

    """

    C50p, C50r, gamma, beta, _, Emax = bis_param

    up = x[3] / C50p
    ur = x[7] / C50r
    Phi = up/(up + ur + 1e-6)
    U50 = 1 - beta * (Phi - Phi**2)
    I = (up + ur)/U50
    dup_dxp = 1/C50p
    dur_dxr = 1/C50r
    dPhi_dxep = dup_dxp * ur/(up + ur + 1e-6)**2
    dPhi_dxer = -dur_dxr * up/((up + ur + 1e-6)**2)
    dU50_dxep = beta*(1 - 2*Phi)*dPhi_dxep
    dU50_dxer = beta*(1 - 2*Phi)*dPhi_dxer
    dI_dxep = (dup_dxp*U50 - dU50_dxep*(up+ur))/U50**2
    dI_dxer = (dur_dxr*U50 - dU50_dxer*(up+ur))/U50**2

    dBIS_dxep = -Emax*gamma*I**(gamma-1)*dI_dxep/(1+I**gamma)**2
    dBIS_dxer = -Emax*gamma*I**(gamma-1)*dI_dxer/(1+I**gamma)**2
    df = np.array([[0, 0, 0, dBIS_dxep[0], 0, 0, 0, dBIS_dxer[0], 1]])

    return df


def compute_bis(xep: float, xer: float, Bis_param: list) -> float:
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
    C50p, C50r, gamma, beta, E0, Emax = Bis_param
    up = xep / C50p
    ur = xer / C50r
    Phi = up/(up + ur + 1e-6)
    U_50 = 1 - beta * (Phi - Phi**2)
    i = (up + ur)/U_50
    BIS = E0 - Emax * i ** gamma / (1 + i ** gamma)
    return BIS


def custom_disturbance(time: float):
    bis_dist = [[0, 0],
                [600, 0],
                [600.1, 20],
                [650, 20],
                [650.1, 10],
                [700, 10],
                [700.1, 20],
                [750, 20],
                [750.1, 10],
                [800, 10],
                [800.1, 5],
                [850, 5],
                [850.1, 0],
                [900, 0],
                [900.1, 10],
                [925, 10],
                [925.1, 0],
                [950, 0],]

    dist = np.interp(time, [x[0] for x in bis_dist], [x[1] for x in bis_dist])
    return [dist, 0, 0]
