from matplotlib import rc
import matplotlib.pyplot as plt
from MEKF_MHE_MPC_study import small_obj
from MEKF_MHE_MPC_study import MHE_param, parem_mekf_mhe, training_patient
import numpy as np
from python_anesthesia_simulator import metrics
import pandas as pd

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

N = 30
R = 30 * np.diag([4, 1])
t_switch = 180
phase = 'induction'

MPC_param = [N, N, R]
parem_mekf_mhe[-1] = t_switch
mhe_nmpc_param = parem_mekf_mhe + MPC_param

df_list = []
TT_list = []
cost_list = []
IAE_list = []
counter = 0
for i in training_patient:
    _, df = small_obj(i, mhe_nmpc_param=mhe_nmpc_param, output='dataframe')
    Time = df['Time'].to_numpy()
    BIS = df['BIS'].to_numpy()
    TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(
        Time, BIS, phase=phase)
    df_list.append(df)
    TT_list.append(TT)
    counter += 1
    cost = compute_cost(df, 'IAE_biased')
    cost_list.append(cost)
    IAE = compute_cost(df, 'IAE')
    IAE_list.append(IAE)
    print(f"Patient {counter}/{len(training_patient)} done")


print(f"Mean training TT : {np.mean(TT_list)*60}")
print(f"Mean training cost : {np.mean(cost_list)}")
print(f"Max training cost : {np.max(cost_list)}")
print(f"Mean training IAE : {np.mean(IAE_list)}")
print(f"Max training IAE : {np.max(IAE_list)}")

# plot all BIS
plt.figure(figsize=(8, 4))
for i in range(len(df_list)):
    plt.plot(df_list[i]['Time'], df_list[i]['BIS'], label=f"Patient {i}")
plt.xlabel("Time (min)")
plt.ylabel("BIS")
# plt.legend()
plt.grid()
plt.savefig(f"./Results_Images/MEKF_MHE_MPC_{phase}_training_manual_R={R[1,1]}.png", dpi=300)
