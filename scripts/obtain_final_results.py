"""
Created on Mon Jan 23 15:03:14 2023

@author: aubouinb
"""
# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors


# plot config
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# First figure: specific case
"""
Created on Tue Jan 24 10:54:59 2023

@author: aubouinb
"""

phase = 'induction'
Number_of_patient = 500
ts = 2
folder_path = './Results_data/'
image_folder_path = './Results_Images/'

try:
    Table_PID = pd.read_csv(f"{folder_path}result_table_PID_{phase}_{Number_of_patient}.csv")
    Table_MHEMPC = pd.read_csv(f"{folder_path}result_table_MHE_NMPC_{phase}_{Number_of_patient}.csv")
    # Table_MMPC = pd.read_csv(f"{folder_path}result_table_MEKF_MHE_NMPC_{phase}_{Number_of_patient}.csv")
    if phase == 'total':
        # Table_MEMPC = pd.read_csv(f"{folder_path}result_table_MEKF_NMPC_{phase}_{Number_of_patient}.csv")
        Table_MMPC = pd.read_csv(f"{folder_path}result_table_MEKF_MHE_NMPC_{phase}_{Number_of_patient}.csv")

    final_table_induction = pd.DataFrame(columns=['Controller', 'IAE', 'TT_mean', 'TT_max', 'BIS_NADIR_mean', 'BIS_NADIR_min',
                                                  'ST10_mean', 'ST10_max', 'ST20_mean', 'ST20_max', 'US_mean', 'US_max'])

    if phase == 'induction':
        final_table_induction['Controller'] = ['PID', 'MHE-NMPC']  # , 'MMM']  # ,'MMPC']  # , 'NMPC', 'MMPC'
        list_data = [Table_PID, Table_MHEMPC]  # , Table_MMPC]
    elif phase == 'total':
        final_table_induction['Controller'] = ['PID', 'MHE-NMPC', 'MMM']  # 'MEKF-MPC'
        list_data = [Table_PID, Table_MHEMPC, Table_MMPC]  # Table_MEMPC

    final_table_induction['IAE'] = [str(df['IAE'][0]) + "$\pm$" + str(df['IAE'][1]) for df in list_data]

    final_table_induction['TT_mean'] = [str(df['TT (min)'][0]) + "$\pm$" + str(df['TT (min)'][1]) for df in list_data]
    final_table_induction['TT_max'] = [str(df['TT (min)'][3]) for df in list_data]

    final_table_induction['BIS_NADIR_mean'] = [
        str(df['BIS_NADIR'][0]) + "$\pm$" + str(df['BIS_NADIR'][1]) for df in list_data]
    final_table_induction['BIS_NADIR_min'] = [str(df['BIS_NADIR'][2]) for df in list_data]

    final_table_induction['ST10_mean'] = [str(df['ST10 (min)'][0]) + "$\pm$" +
                                          str(df['ST10 (min)'][1]) for df in list_data]
    final_table_induction['ST10_max'] = [str(df['ST10 (min)'][3]) for df in list_data]

    final_table_induction['ST20_mean'] = [str(df['ST20 (min)'][0]) + "$\pm$" +
                                          str(df['ST20 (min)'][1]) for df in list_data]
    final_table_induction['ST20_max'] = [str(df['ST20 (min)'][3]) for df in list_data]

    final_table_induction['US_mean'] = [str(df['US'][0]) + "$\pm$" + str(df['US'][1]) for df in list_data]
    final_table_induction['US_max'] = [str(df['US'][3]) for df in list_data]

    styler = final_table_induction.style
    styler.hide(axis='index')
    styler.format(precision=2)
    print(styler.to_latex())

    if phase == 'total':
        final_table_maintenance = pd.DataFrame(columns=['Controller', 'IAE', 'TTp_mean', 'BIS_NADIRp_mean', 'BIS_NADIRp_min',
                                                        'TTn_mean', 'BIS_NADIRn_mean', 'BIS_NADIRn_min'])

        if phase == 'induction':
            final_table_maintenance['Controller'] = ['PID', 'MHE-NMPC']  # ,'MMPC']  # , 'NMPC', 'MMPC'
            list_data = [Table_PID, Table_MHEMPC]
        elif phase == 'total':
            final_table_maintenance['Controller'] = ['PID', 'MHE-NMPC', 'MMM']  # 'MEKF-MPC'
            list_data = [Table_PID, Table_MHEMPC, Table_MMPC]  # Table_MEMPC

        final_table_maintenance['IAE'] = [str(df['IAE_maintenance'][0]) + "$\pm$" +
                                          str(df['IAE_maintenance'][1]) for df in list_data]

        final_table_maintenance['TTp_mean'] = [
            str(df['TTp (min)'][0]) + "$\pm$" + str(df['TTp (min)'][1]) for df in list_data]
        # final_table_maintenance['TTp_max'] = [str(df['TTp (min)'][3]) for df in list_data]

        final_table_maintenance['BIS_NADIRp_mean'] = [
            str(df['BIS_NADIRp'][0]) + "$\pm$" + str(df['BIS_NADIRp'][1]) for df in list_data]
        final_table_maintenance['BIS_NADIRp_min'] = [str(df['BIS_NADIRp'][2]) for df in list_data]

        final_table_maintenance['TTn_mean'] = [
            str(df['TTn (min)'][0]) + "$\pm$" + str(df['TTn (min)'][1]) for df in list_data]
        # final_table_maintenance['TTn_max'] = [str(df['TTn (min)'][3]) for df in list_data]

        final_table_maintenance['BIS_NADIRn_mean'] = [
            str(df['BIS_NADIRn'][0]) + "$\pm$" + str(df['BIS_NADIRn'][1]) for df in list_data]
        final_table_maintenance['BIS_NADIRn_min'] = [str(df['BIS_NADIRn'][2]) for df in list_data]

        styler = final_table_maintenance.style
        styler.hide(axis='index')
        styler.format(precision=2)
        print(styler.to_latex())

except FileNotFoundError:
    print("No data available to construct table")


# get data


bool_PID = False
bool_NMPC = False
bool_MMPC = False
bool_MHEMPC = False
try:
    Data_PID = pd.read_csv(f"{folder_path}PID_{phase}_{Number_of_patient}.csv")
    bool_PID = True
except FileNotFoundError:
    print("No data available for PID")

# try:
#     Data_NMPC = pd.read_csv(f"{folder_path}MEKF_NMPC_{phase}_{Number_of_patient}.csv")
#     bool_NMPC = True
# except FileNotFoundError:
#     print("No data available for NMPC")

try:
    Data_MMPC = pd.read_csv(f"{folder_path}MEKF_MHE_NMPC_{phase}_{Number_of_patient}.csv")
    bool_MMPC = True
except FileNotFoundError:
    print("No data available for MMPC")

try:
    Data_MHEMPC = pd.read_csv(f"{folder_path}MHE_NMPC_{phase}_{Number_of_patient}.csv")
    bool_MHEMPC = True
except FileNotFoundError:
    print("No data available for MHEMPC")


# find Patient with minimum BIS for each controller
Patient_id_min_PID_induction = 0
Patient_id_min_NMPC_induction = 0
Patient_id_min_MMPC_induction = 0
Patient_id_min_MHEMPC_induction = 0

Patient_id_min_PID_maintenance = 0
Patient_id_min_NMPC_maintenance = 0
Patient_id_min_MMPC_maintenance = 0
Patient_id_min_MHEMPC_maintenance = 0

for patient_id in range(1, Number_of_patient):
    if bool_PID:
        if np.min(Data_PID[str(patient_id)+'_BIS'].loc[0:9*60//ts]) < np.min(Data_PID[str(Patient_id_min_PID_induction)+'_BIS'].loc[0:9*60//ts]):
            Patient_id_min_PID_induction = patient_id
        if np.min(Data_PID[str(patient_id)+'_BIS'].loc[9*60//ts:]) < np.min(Data_PID[str(Patient_id_min_PID_maintenance)+'_BIS'].loc[9*60//ts:]):
            Patient_id_min_PID_maintenance = patient_id

    if bool_NMPC:
        if np.min(Data_NMPC[str(patient_id)+'_BIS'].loc[0:9*60//ts]) < np.min(Data_NMPC[str(Patient_id_min_NMPC_induction)+'_BIS'].loc[0:9*60//ts]):
            Patient_id_min_NMPC_induction = patient_id
        if np.min(Data_NMPC[str(patient_id)+'_BIS'].loc[9*60//ts:]) < np.min(Data_NMPC[str(Patient_id_min_NMPC_maintenance)+'_BIS'].loc[9*60//ts:]):
            Patient_id_min_NMPC_maintenance = patient_id

    if bool_MMPC:
        if np.min(Data_MMPC[str(patient_id)+'_BIS'].loc[0:9*60//ts]) < np.min(Data_MMPC[str(Patient_id_min_MMPC_induction)+'_BIS'].loc[0:9*60//ts]):
            Patient_id_min_MMPC_induction = patient_id
        if np.min(Data_MMPC[str(patient_id)+'_BIS'].loc[9*60//ts:]) < np.min(Data_MMPC[str(Patient_id_min_MMPC_maintenance)+'_BIS'].loc[9*60//ts:]):
            Patient_id_min_MMPC_maintenance = patient_id
    if bool_MHEMPC:
        if np.min(Data_MHEMPC[str(patient_id)+'_BIS'].loc[0:9*60//ts]) < np.min(Data_MHEMPC[str(Patient_id_min_MHEMPC_induction)+'_BIS'].loc[0:9*60//ts]):
            Patient_id_min_MHEMPC_induction = patient_id
        if np.min(Data_MHEMPC[str(patient_id)+'_BIS'].loc[9*60//ts:]) < np.min(Data_MHEMPC[str(Patient_id_min_MHEMPC_maintenance)+'_BIS'].loc[9*60//ts:]):
            Patient_id_min_MHEMPC_maintenance = patient_id

print(f"worst patient for EKF-MPC: {Patient_id_min_NMPC_induction}")
print(f"worst patient for MHE-MPC: {Patient_id_min_MHEMPC_induction}")
# if phase == 'induction':
#     Patient_id_min_PID = Patient_id_min_MHEMPC
# if phase == 'total':
#     Patient_id_min_PID = Patient_id_min_MMPC
#     Patient_id_min_MHEMPC = Patient_id_min_MMPC
if bool_PID:
    BIS_PID = Data_PID[str(Patient_id_min_PID_induction)+'_BIS']
if bool_NMPC:
    BIS_NMPC = Data_NMPC[str(Patient_id_min_NMPC_induction)+'_BIS']
if bool_MMPC:
    BIS_MMPC = Data_MMPC[str(Patient_id_min_MMPC_induction)+'_BIS']
if bool_MHEMPC:
    BIS_MHEMPC = Data_MHEMPC[str(Patient_id_min_MHEMPC_induction)+'_BIS']

if bool_PID:
    Time_PID = Data_PID['0_Time'] / 60
if bool_NMPC:
    Time_MPC = Data_NMPC['0_Time'] / 60
elif bool_NMPC:
    Time_MPC = Data_MMPC['0_Time'] / 60
elif bool_MHEMPC:
    Time_MPC = Data_MHEMPC['0_Time'] / 60

if bool_PID:
    Up_PID = Data_PID[str(Patient_id_min_PID_induction)+'_u_propo']
    Ur_PID = Data_PID[str(Patient_id_min_PID_induction)+'_u_remi']
if bool_NMPC:
    Up_NMPC = Data_NMPC[str(Patient_id_min_NMPC_induction)+'_u_propo']
    Ur_NMPC = Data_NMPC[str(Patient_id_min_NMPC_induction)+'_u_remi']
if bool_MMPC:
    Up_MMPC = Data_MMPC[str(Patient_id_min_MMPC_induction)+'_u_propo']
    Ur_MMPC = Data_MMPC[str(Patient_id_min_MMPC_induction)+'_u_remi']
if bool_MHEMPC:
    Up_MHEMPC = Data_MHEMPC[str(Patient_id_min_MHEMPC_induction)+'_u_propo']
    Ur_MHEMPC = Data_MHEMPC[str(Patient_id_min_MHEMPC_induction)+'_u_remi']


# %% Create BIS figure
if phase == 'total':
    fig, ax = plt.subplots(figsize=(10, 4))
else:
    fig, ax = plt.subplots(2, 1)
if bool_PID:
    ax[0].plot(Time_PID, BIS_PID, label='PID')
if bool_NMPC:
    ax[0].plot(Time_MPC, BIS_NMPC, label='MEKF-MPC')
if bool_MHEMPC:
    ax[0].plot(Time_MPC, BIS_MHEMPC, label='MHE-MPC', color=mcolors.TABLEAU_COLORS['tab:red'])
if bool_MMPC:
    ax[0].plot(Time_MPC, BIS_MMPC, label='MMM', color=mcolors.TABLEAU_COLORS['tab:green'])
ax[0].grid(linewidth=0.4)
ax[0].legend(fontsize=13)

# ax.set_yticks(list(ax.get_yticks()) + [int(BIS_PID[0])])
ax[0].set_ylim([16, 102])
# ax.set_xlim([-0.2, 10.2])
# ax.set_xlabel('Time (min)', fontsize=13)
ax[0].set_ylabel('BIS', fontsize=13)
# ygridlines = ax.get_ygridlines()
# gridline_of_interest = ygridlines[-1]
# gridline_of_interest.set_visible(False)
plt.draw()

# save it
# savepath = image_folder_path + f"worst_bis_induction.pdf"
# plt.savefig(savepath, bbox_inches='tight', format='pdf')
# plt.show()

# Create drug rates figures
linewidth = 1.3

if bool_PID:
    ax[1].plot(Time_PID, Up_PID, label='$u_p$ PID',
               linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:blue'])
    ax[1].plot(Time_PID, Ur_PID, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:blue'],
               label='$u_r$  PID')
if bool_NMPC:
    ax[1].plot(Time_MPC, Up_NMPC, label='Propofol MEKF-MPC (mg/s)',
               linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:orange'])
    ax[1].plot(Time_MPC, Ur_NMPC, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:orange'],
               label='Remifentanil MEKF-MPC (µg/s)')
if bool_MMPC:
    ax[1].plot(Time_MPC, Up_MMPC, label='Propofol MMM (mg/s)',
               linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:green'])
    ax[1].plot(Time_MPC, Ur_MMPC, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:green'],
               label='Remifentanil MMM (µg/s)')
if bool_MHEMPC:
    ax[1].plot(Time_MPC, Up_MHEMPC, label='$u_p$ MHE-MPC',
               linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:red'])
    ax[1].plot(Time_MPC, Ur_MHEMPC, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:red'],
               label='$u_r$ MHE-MPC')

ax[1].grid(linewidth=0.4)
ax[1].legend(fontsize=13)
# plt.xlim([-0.2, 10.2])
ax[1].set_xlabel('Time (min)', fontsize=13)
ax[1].set_ylabel('Drug rates', fontsize=13)
# use log scale for y axis
# ax[1].set_yscale('log')

plt.draw()

# save it
savepath = image_folder_path + f"worst_bis_inputs_induction.pdf"
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()

if phase == 'total':
    if bool_PID:
        BIS_PID = Data_PID[str(Patient_id_min_PID_maintenance)+'_BIS']
    if bool_NMPC:
        BIS_NMPC = Data_NMPC[str(Patient_id_min_NMPC_maintenance)+'_BIS']
    if bool_MMPC:
        BIS_MMPC = Data_MMPC[str(Patient_id_min_MMPC_maintenance)+'_BIS']
    if bool_MHEMPC:
        BIS_MHEMPC = Data_MHEMPC[str(Patient_id_min_MHEMPC_maintenance)+'_BIS']

    if bool_PID:
        Time_PID = Data_PID['0_Time'] / 60
    if bool_NMPC:
        Time_MPC = Data_NMPC['0_Time'] / 60
    elif bool_NMPC:
        Time_MPC = Data_MMPC['0_Time'] / 60
    elif bool_MHEMPC:
        Time_MPC = Data_MHEMPC['0_Time'] / 60

    if bool_PID:
        Up_PID = Data_PID[str(Patient_id_min_PID_maintenance)+'_u_propo']
        Ur_PID = Data_PID[str(Patient_id_min_PID_maintenance)+'_u_remi']
    if bool_NMPC:
        Up_NMPC = Data_NMPC[str(Patient_id_min_NMPC_maintenance)+'_u_propo']
        Ur_NMPC = Data_NMPC[str(Patient_id_min_NMPC_maintenance)+'_u_remi']
    if bool_MMPC:
        Up_MMPC = Data_MMPC[str(Patient_id_min_MMPC_maintenance)+'_u_propo']
        Ur_MMPC = Data_MMPC[str(Patient_id_min_MMPC_maintenance)+'_u_remi']
    if bool_MHEMPC:
        Up_MHEMPC = Data_MHEMPC[str(Patient_id_min_MHEMPC_maintenance)+'_u_propo']
        Ur_MHEMPC = Data_MHEMPC[str(Patient_id_min_MHEMPC_maintenance)+'_u_remi']

    # %% Create BIS figure
    ax: matplotlib.pyplot.Axes
    if phase == 'total':
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig, ax = plt.subplots()
    if bool_PID:
        ax.plot(Time_PID, BIS_PID, label='PID')
    if bool_NMPC:
        ax.plot(Time_MPC, BIS_NMPC, label='MEKF-MPC')
    if bool_MHEMPC:
        ax.plot(Time_MPC, BIS_MHEMPC, label='MHE-MPC', color=mcolors.TABLEAU_COLORS['tab:red'])
    if bool_MMPC:
        ax.plot(Time_MPC, BIS_MMPC, label='MMM', color=mcolors.TABLEAU_COLORS['tab:green'])
    ax.grid(linewidth=0.4)
    ax.legend(fontsize=13)

    # ax.set_yticks(list(ax.get_yticks()) + [int(BIS_PID[0])])
    ax.set_ylim([16, 102])
    # ax.set_xlim([-0.2, 10.2])
    ax.set_xlabel('Time (min)', fontsize=13)
    ax.set_ylabel('BIS', fontsize=13)
    # ygridlines = ax.get_ygridlines()
    # gridline_of_interest = ygridlines[-1]
    # gridline_of_interest.set_visible(False)
    plt.draw()

    # save it
    savepath = image_folder_path + f"worst_bis_maintenance.pdf"
    plt.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()

    # Create drug rates figures
    linewidth = 1.3
    if phase == 'total':
        plt.figure(figsize=(10, 4))
    if bool_PID:
        plt.plot(Time_PID, Up_PID, label='Propofol PID (mg/s)',
                 linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:blue'])
        plt.plot(Time_PID, Ur_PID, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:blue'],
                 label='Remifentanil PID (µg/s)')
    if bool_NMPC:
        plt.plot(Time_MPC, Up_NMPC, label='Propofol MEKF-MPC (mg/s)',
                 linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:orange'])
        plt.plot(Time_MPC, Ur_NMPC, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:orange'],
                 label='Remifentanil MEKF-MPC (µg/s)')
    if bool_MMPC:
        plt.plot(Time_MPC, Up_MMPC, label='Propofol MMPC (mg/s)',
                 linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:green'])
        plt.plot(Time_MPC, Ur_MMPC, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:green'],
                 label='Remifentanil MMPC (µg/s)')
    if bool_MHEMPC:
        plt.plot(Time_MPC, Up_MHEMPC, label='Propofol MHE-MPC (mg/s)',
                 linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:red'])
        plt.plot(Time_MPC, Ur_MHEMPC, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:red'],
                 label='Remifentanil MHE-MPC (µg/s)')

    plt.grid(linewidth=0.4)
    plt.legend(fontsize=13)
    # plt.xlim([-0.2, 10.2])
    plt.xlabel('Time (min)', fontsize=13)
    plt.draw()

    # save it
    savepath = image_folder_path + f"worst_bis_inputs_maintenance.pdf"
    plt.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()

# %% Second figure: mean values
transparency = 0.3

if phase == 'total':
    plt.figure(figsize=(10, 4))

if bool_PID:
    BIS_PID = Data_PID.loc[:, Data_PID.columns.str.contains('BIS')]
    mean_BIS_PID = BIS_PID.mean(axis=1)
    std_BIS_PID = BIS_PID.std(axis=1)
    plt.fill_between(Time_PID, mean_BIS_PID-std_BIS_PID, mean_BIS_PID+std_BIS_PID,
                     alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:blue'])

if bool_NMPC:
    BIS_NMPC = Data_NMPC.loc[:, Data_NMPC.columns.str.contains('BIS')]
    mean_BIS_NMPC = BIS_NMPC.mean(axis=1)
    std_BIS_NMPC = BIS_NMPC.std(axis=1)
    plt.fill_between(Time_MPC, mean_BIS_NMPC-std_BIS_NMPC, mean_BIS_NMPC+std_BIS_NMPC,
                     alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:orange'])

if bool_MHEMPC:
    BIS_MHEMPC = Data_MHEMPC.loc[:, Data_MHEMPC.columns.str.contains('BIS')]
    mean_BIS_MHEMPC = BIS_MHEMPC.mean(axis=1)
    std_BIS_MHEMPC = BIS_MHEMPC.std(axis=1)
    plt.fill_between(Time_MPC, mean_BIS_MHEMPC-std_BIS_MHEMPC, mean_BIS_MHEMPC+std_BIS_MHEMPC,
                     alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:red'])
if bool_MMPC:
    BIS_MMPC = Data_MMPC.loc[:, Data_MMPC.columns.str.contains('BIS')]
    mean_BIS_MMPC = BIS_MMPC.mean(axis=1)
    std_BIS_MMPC = BIS_MMPC.std(axis=1)
    plt.fill_between(Time_MPC, mean_BIS_MMPC-std_BIS_MMPC, mean_BIS_MMPC+std_BIS_MMPC,
                     alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:green'])


if bool_PID:
    plt.plot(Time_PID, mean_BIS_PID-std_BIS_PID, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:blue'], linewidth=1)
if bool_NMPC:
    plt.plot(Time_MPC, mean_BIS_NMPC-std_BIS_NMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:orange'], linewidth=1)
if bool_MHEMPC:
    plt.plot(Time_MPC, mean_BIS_MHEMPC-std_BIS_MHEMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:red'], linewidth=1)
if bool_MMPC:
    plt.plot(Time_MPC, mean_BIS_MMPC-std_BIS_MMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:green'], linewidth=1)

if bool_PID:
    plt.plot(Time_PID, mean_BIS_PID+std_BIS_PID, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:blue'], linewidth=1)
if bool_NMPC:
    plt.plot(Time_MPC, mean_BIS_NMPC+std_BIS_NMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:orange'], linewidth=1)
if bool_MHEMPC:
    plt.plot(Time_MPC, mean_BIS_MHEMPC+std_BIS_MHEMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:red'], linewidth=1)
if bool_MMPC:
    plt.plot(Time_MPC, mean_BIS_MMPC+std_BIS_MMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:green'], linewidth=1)

if bool_PID:
    plt.plot(Time_PID, mean_BIS_PID, label='PID', color=mcolors.TABLEAU_COLORS['tab:blue'])
if bool_NMPC:
    plt.plot(Time_MPC, mean_BIS_NMPC, label='MEKF-MPC', color=mcolors.TABLEAU_COLORS['tab:orange'])
if bool_MHEMPC:
    plt.plot(Time_MPC, mean_BIS_MHEMPC, label='MHE-MPC', color=mcolors.TABLEAU_COLORS['tab:red'])
if bool_MMPC:
    plt.plot(Time_MPC, mean_BIS_MMPC, label='MMPC', color=mcolors.TABLEAU_COLORS['tab:green'])

plt.grid(linewidth=0.4)
plt.legend(fontsize=13)
plt.xlabel('Time (min)', fontsize=13)
plt.ylabel('BIS', fontsize=13)
plt.ylim([40, 100])
# plt.xlim([-0.2, 10.2])
plt.draw()

# save it
savepath = image_folder_path + "BIS_mean_case_" + phase + ".pdf"
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()

# %% illustrate cost choice

# time = np.linspace(0, 15*60, 100)

# gamma = 1.e-2
# theta = [gamma, 800, 100, 0.005]*4
# theta[4] = gamma/100
# theta[12] = gamma/100
# theta[13] = 300
# theta[15] = 0.05


# Q8 = theta[0] + theta[1]*np.exp(-theta[2]*np.exp(-theta[3]*time))
# Q9 = theta[4] + theta[5]*np.exp(-theta[6]*np.exp(-theta[7]*time))
# Q10 = theta[8] + theta[9]*np.exp(-theta[10]*np.exp(-theta[11]*time))
# Q11 = theta[12] + theta[13]*(1-np.exp(-theta[14]*np.exp(-theta[15]*time)))

# plt.plot(time/60, Q8, label='Q8')
# plt.plot(time/60, Q9, label='Q9')
# plt.plot(time/60, Q10, label='Q10')
# plt.plot(time/60, Q11, label='Q11')
# plt.legend()
# plt.yscale('log')
# plt.grid()
# # set aspect ratio to 1
# ax = plt.gca()
# ratio = 0.9
# # x_left, x_right = ax.get_xlim()
# # y_low, y_high = ax.get_ylim()
# # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
# plt.plot()

# # save it
# savepath = image_folder_path + "cost.pdf"
# plt.savefig(savepath, bbox_inches='tight', format='pdf')
# plt.show()
