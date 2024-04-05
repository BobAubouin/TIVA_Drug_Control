import numpy as np
import pandas as pd
import os

from close_loop_anesth.loop import perform_simulation

np.random.seed(0)
training_patient = np.random.randint(0, 500, size=16)


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def compute_cost(df: pd.DataFrame, type: str) -> float:
    """Compute the cost of the simulation.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of the simulation.
    type : str
        type of the cost. can be 'IAE' or 'TT'.

    Returns
    -------
    float
        cost of the simulation.
    """
    if type == 'IAE':
        cost = np.sum((df['BIS'] - 50)**2, axis=0)
    elif type == 'IAE_biased':
        mask = df['BIS'] > 50
        cost = np.sum((df['BIS'] - 50)**3 * mask + (df['BIS'] - 50)**4 * (~mask), axis=0)
    elif type == 'TT':
        for i in range(len(df['BIS'])):
            if df['BIS'].iloc[i] < 60:
                break
        cost = (df['Time'].iloc[i] - 101)**2
    return cost


def random_simu(caseid: int,
                control_type: str,
                control_param: dict,
                estim_param: dict,
                output: str = 'IAE',
                phase: str = 'induction',
                cost_choice: str = 'cost'):
    np.random.seed(caseid)
    print(caseid)
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    with suppress_stdout_stderr():
        df_results = perform_simulation([age, height, weight, gender],
                                        phase,
                                        control_type=control_type,
                                        control_param=control_param,
                                        estim_param=estim_param,
                                        random_bool=[True, True])
    if output == 'cost':
        cost = compute_cost(df_results, cost_choice)
        return cost
    elif output == 'dataframe':
        df_results.insert(0, 'caseid', caseid)
        return df_results
    else:
        return
