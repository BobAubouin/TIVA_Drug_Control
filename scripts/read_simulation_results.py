"""
Created on Tue Nov 22 17:16:03 2022

@author: aubouinb
"""
# Standard import

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from python_anesthesia_simulator import metrics


Number_of_patient = 500
phase = 'induction'

# choose the file to read, NMPC and MMPC have a sample time of 2s, PID of 1s.
title = 'NMPC'
title = 'multi_NMPC'
title = 'MHE_NMPC'
# title = 'MPC_lin'
title = 'PID'
# title = 'NMPC_int_induction'
# title = 'multi_MMPC_int'
if title == 'PID':
    ts = 1
else:
    ts = 2
Data = pd.read_csv("./Results_data/result_" + title + "_n=" + str(Number_of_patient) + '.csv')


if phase == 'induction':
    IAE_list = []
    TT_list = []
    ST10_list = []
    ST20_list = []
    US_list = []
    BIS_NADIR_list = []
elif phase == 'maintenance':
    TTp_list = []
    TTn_list = []
    BIS_NADIRp_list = []
    BIS_NADIRn_list = []


for i in range(Number_of_patient):  # Number_of_patient
    # for i in range(108, 109):
    print(i)

    BIS = Data[str(i) + '_BIS']
    Time = np.arange(0, len(BIS)) * ts / 60
    plt.plot(Time, BIS, linewidth=0.5)

    if phase == 'induction':
        TT, BIS_NADIR, ST10, ST20, US = metrics.compute_control_metrics(
            np.arange(0, len(BIS)) * ts, BIS, phase=phase)
        TT_list.append(TT)
        BIS_NADIR_list.append(BIS_NADIR)
        ST10_list.append(ST10)
        ST20_list.append(ST20)
        US_list.append(US)
    elif phase == 'maintenance':
        TTp, BIS_NADIRp, TTn, BIS_NADIRn = metrics.compute_control_metrics(
            np.arange(0, len(BIS)) * ts, BIS, phase=phase)
        TTp_list.append(TTp)
        TTn_list.append(TTn)
        BIS_NADIRp_list.append(BIS_NADIRp)
        BIS_NADIRn_list.append(BIS_NADIRn)

plt.xlabel('Time (min)')
plt.ylabel('BIS')
plt.grid()
plt.savefig('./Results_Images/BIS_' + title + '_n=' + str(Number_of_patient) + '.pdf')
plt.show()

result_table = pd.DataFrame()
result_table.insert(len(result_table.columns), "", ['mean', 'std', 'min', 'max'])
if phase == 'induction':
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
elif phase == 'maintenance':
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


print('\n')
styler = result_table.style
styler.hide(axis='index')
styler.format(precision=2)
print(styler.to_latex())

result_table.to_csv("./Results_data/result_table" + title + "_n=" + str(Number_of_patient))
