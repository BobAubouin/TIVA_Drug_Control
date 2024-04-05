"""
Created on Tue Nov 22 17:16:03 2022

@author: aubouinb
"""
# Standard import

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from python_anesthesia_simulator import metrics


Number_of_patient = 500
phase = 'induction'

np.random.seed(0)
training_patient = np.random.randint(0, 500, size=16)

# choose the file to read, NMPC and MMPC have a sample time of 2s, PID of 1s.

# title = 'PID'
# title = 'MEKF_NMPC'
# title = 'EKF_NMPC'
title = 'MHE_NMPC'
# title = 'MEKF_MHE_NMPC'
print(f"Reading {title} results")
if title == 'PID':
    ts = 2
else:
    ts = 2
Data = pd.read_csv(f"./Results_data/{title}_{phase}_{Number_of_patient}.csv")


if phase == 'induction':
    IAE_list = []
    TT_list = []
    ST10_list = []
    ST20_list = []
    US_list = []
    BIS_NADIR_list = []
elif phase == 'total':
    IAE_list = []
    TT_list = []
    ST10_list = []
    ST20_list = []
    US_list = []
    BIS_NADIR_list = []
    TTp_list = []
    TTn_list = []
    BIS_NADIRp_list = []
    BIS_NADIRn_list = []

BIS_data = Data[[f"{i}_BIS" for i in range(Number_of_patient)]].to_numpy()
BIS_data_training = Data[[f"{i}_BIS" for i in training_patient]].to_numpy()
Time = np.arange(0, len(Data)) * ts / 60

# IAE = np.sum(np.abs(BIS_data_training - 50)**2 * ts, axis=0)

# # create a metric which penalize two time more BIS under 50 that upside
# IAE = np.sum((BIS_data_training - 50)**2*(1+((BIS_data_training - 50) < 0)) * ts, axis=0)
# IAE = []
# for i in range(len(training_patient)):
#     mask = BIS_data_training[:, i] > 50
#     IAE.append(np.sum((BIS_data_training[:, i] - 50)**3 * mask + (BIS_data_training[:, i] - 50)**4 * (~mask), axis=0))


def compute_TT(BIS: np.array) -> float:
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
    for i in range(len(BIS)):
        if BIS[i] < 55:
            break
    return i*ts


TT_training = []
for i in range(len(training_patient)):
    TT_training.append(compute_TT(BIS_data_training[:, i]))
print(f"Training TT : {np.mean(TT_training)}")

if phase == 'induction':
    IAE = np.sum(np.abs(BIS_data_training - 50) * ts, axis=0)
elif phase == 'total':
    IAE = np.sum(np.abs(BIS_data_training[0:10*60//ts] - 50) * ts, axis=0)
    IAE_maintenance = np.sum(np.abs(BIS_data_training[10*60//ts:] - 50) * ts, axis=0)

print(f"Training IAE : {np.max(IAE)}")
print(f"Patient training with max IAE : {training_patient[np.argmax(IAE)]}")


plt.subplot(2, 1, 1)
plt.title(title + ' ' + phase)
plt.plot(Time, BIS_data, linewidth=1, color='b')
plt.plot(Time, BIS_data_training, linewidth=1, color='green', alpha=1)
plt.plot(Time, BIS_data_training[:, np.argmax(IAE)], linewidth=1, color='orange', alpha=1)
plt.plot(Time, np.nanmean(BIS_data, axis=1), linewidth=1, color='r')
plt.ylabel('BIS')
plt.grid()
Up_data = Data[[f"{i}_u_propo" for i in range(Number_of_patient)]].to_numpy()
Ur_data = Data[[f"{i}_u_remi" for i in range(Number_of_patient)]].to_numpy()

plt.subplot(2, 1, 2)
plt.plot(Time, Up_data, linewidth=0.5, color='b', alpha=0.1)
plt.plot(Time, Ur_data, linewidth=0.5, color='r', alpha=0.1)
plt.plot(Time, np.nanmean(Up_data, axis=1), linewidth=1, color='b', label='Propofol')
plt.plot(Time, np.nanmean(Ur_data, axis=1), linewidth=1, color='r', label='Remifentanil')
plt.ylabel('Inputs')
plt.grid()
plt.xlabel('Time (min)')
plt.legend()
plt.savefig('./Results_Images/BIS_' + title + '_n=' + str(Number_of_patient) + '.pdf')
plt.show()


for i in tqdm(range(Number_of_patient)):  # Number_of_patient

    BIS = Data[str(i) + '_BIS']
    Time = Data[str(i) + '_Time']

    if phase == 'induction':
        TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(
            Time, BIS, phase=phase)
        TT_list.append(TT)
        BIS_NADIR_list.append(BIS_NADIR)
        ST10_list.append(ST10)
        ST20_list.append(ST20)
        US_list.append(US)
    elif phase == 'total':
        TT, BIS_NADIR, ST10, ST20, US, TTp, BIS_NADIRp, TTn, BIS_NADIRn = metrics.compute_control_metrics(
            Time, BIS, phase=phase, start_step=10*60, end_step=15*60)
        if np.isnan(ST10):
            ST10 = 10
        if np.isnan(ST20):
            ST20 = 10

        TT_list.append(TT)
        BIS_NADIR_list.append(BIS_NADIR)
        ST10_list.append(ST10)
        ST20_list.append(ST20)
        US_list.append(US)
        TTp_list.append(TTp)
        TTn_list.append(TTn)
        BIS_NADIRp_list.append(BIS_NADIRp)
        BIS_NADIRn_list.append(BIS_NADIRn)

# percentage of patient with BIS value below 40
print(
    f"Percentage of patient with BIS value below 40 : {np.sum(np.nanmin(BIS_data, axis=0) < 40)/Number_of_patient*100}%")


result_table = pd.DataFrame()
result_table.insert(len(result_table.columns), "", ['mean', 'std', 'min', 'max'])
if phase == 'induction' or phase == 'total':
    result_table.insert(len(result_table.columns),
                        "TT (min)", [np.round(np.nanmean(TT_list), 2),
                                     np.round(np.nanstd(TT_list), 2),
                                     np.round(np.nanmin(TT_list), 2),
                                     np.round(np.nanmax(TT_list), 2)])

    result_table.insert(len(result_table.columns),
                        "BIS_NADIR", [np.round(np.nanmean(BIS_NADIR_list), 2),
                                      np.round(np.nanstd(BIS_NADIR_list), 2),
                                      np.round(np.nanmin(BIS_NADIR_list), 2),
                                      np.round(np.nanmax(BIS_NADIR_list), 2)])

    result_table.insert(len(result_table.columns),
                        "ST10 (min)", [np.round(np.nanmean(ST10_list), 2),
                                       np.round(np.nanstd(ST10_list), 2),
                                       np.round(np.nanmin(ST10_list), 2),
                                       np.round(np.nanmax(ST10_list), 2)])

    result_table.insert(len(result_table.columns),
                        "ST20 (min)", [np.round(np.nanmean(ST20_list), 2),
                                       np.round(np.nanstd(ST20_list), 2),
                                       np.round(np.nanmin(ST20_list), 2),
                                       np.round(np.nanmax(ST20_list), 2)])

    result_table.insert(len(result_table.columns),
                        "US", [np.round(np.nanmean(US_list), 2),
                               np.round(np.nanstd(US_list), 2),
                               np.round(np.nanmin(US_list), 2),
                               np.round(np.nanmax(US_list), 2)])
    result_table.insert(len(result_table.columns),
                        "IAE", [np.round(np.nanmean(IAE), 0),
                                np.round(np.nanstd(IAE), 0),
                                np.round(np.nanmin(IAE), 0),
                                np.round(np.nanmax(IAE), 0)])

if phase == 'maintenance' or phase == 'total':
    result_table.insert(len(result_table.columns),
                        "TTp (min)", [np.round(np.nanmean(TTp_list), 2),
                                      np.round(np.nanstd(TTp_list), 2),
                                      np.round(np.nanmin(TTp_list), 2),
                                      np.round(np.nanmax(TTp_list), 2)])

    result_table.insert(len(result_table.columns),
                        "BIS_NADIRp", [np.round(np.nanmean(BIS_NADIRp_list), 2),
                                       np.round(np.nanstd(BIS_NADIRp_list), 2),
                                       np.round(np.nanmin(BIS_NADIRp_list), 2),
                                       np.round(np.nanmax(BIS_NADIRp_list), 2)])

    result_table.insert(len(result_table.columns),
                        "TTn (min)", [np.round(np.nanmean(TTn_list), 2),
                                      np.round(np.nanstd(TTn_list), 2),
                                      np.round(np.nanmin(TTn_list), 2),
                                      np.round(np.nanmax(TTn_list), 2)])

    result_table.insert(len(result_table.columns),
                        "BIS_NADIRn", [np.round(np.nanmean(BIS_NADIRn_list), 2),
                                       np.round(np.nanstd(BIS_NADIRn_list), 2),
                                       np.round(np.nanmin(BIS_NADIRn_list), 2),
                                       np.round(np.nanmax(BIS_NADIRn_list), 2)])
    result_table.insert(len(result_table.columns),
                        "IAE_maintenance", [np.round(np.nanmean(IAE_maintenance), 0),
                                            np.round(np.nanstd(IAE_maintenance), 0),
                                            np.round(np.nanmin(IAE_maintenance), 0),
                                            np.round(np.nanmax(IAE_maintenance), 0)])


print('\n')
styler = result_table.style
styler.hide(axis='index')
styler.format(precision=2)
print(styler.to_latex())

result_table.to_csv(f"./Results_data/result_table_{title}_{phase}_{Number_of_patient}.csv")
