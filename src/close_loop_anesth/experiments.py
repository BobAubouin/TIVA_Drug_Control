import numpy as np
import pandas as pd
import os

from close_loop_anesth.loop import perform_simulation

np.random.seed(0)
training_patient = np.random.randint(0, 500, size=16)


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
    elif type == 'IAE_biased_normal':
        TIME_MAINTENANCE = 599
        bis_induction = df[df.Time<TIME_MAINTENANCE].BIS
        bis_maintenance = df[df.Time>=TIME_MAINTENANCE].BIS
        mask_induction = bis_induction > 50
        biased_cost = np.sum((bis_induction - 50)**3 * mask_induction + (bis_induction - 50)**4 * (~mask_induction), axis=0)
        normal_cost = np.sum((bis_maintenance - 50)**4, axis=0)
        cost = biased_cost + normal_cost
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
    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

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
