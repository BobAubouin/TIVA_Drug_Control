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
import tikzplotlib

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

phase = 'total'

folder_path = './Results_data/'
image_folder_path = './Results_Images/'

try:
    Table_PID = pd.read_csv(folder_path + 'result_table_PID_' + phase + '_n=500')
    Table_NMPC = pd.read_csv(folder_path + 'result_table_NMPC_' + phase + '_n=500')
    Table_MMPC = pd.read_csv(folder_path + 'result_table_multi_' + phase + '_NMPC_n=500')
    Table_MHEMPC = pd.read_csv(folder_path + 'result_table_MHE_' + phase + '_NMPC_n=500')

    final_table = pd.DataFrame(columns=['Controller', 'TT_mean', 'TT_max', 'BIS_NADIR_mean', 'BIS_NADIR_min',
                                        'ST10_mean', 'ST10_max', 'ST20_mean', 'ST20_max', 'US_mean', 'US_max'])

    final_table['Controller'] = ['PID', 'NMPC', 'MMPC', 'MHE_NMPC']

    list_data = [Table_PID, Table_NMPC, Table_MMPC, Table_MHEMPC]

    final_table['TT_mean'] = [str(df['TT (min)'][0]) + "$\pm$" + str(df['TT (min)'][1]) for df in list_data]
    final_table['TT_max'] = [str(df['TT (min)'][3]) for df in list_data]

    final_table['BIS_NADIR_mean'] = [str(df['BIS_NADIR'][0]) + "$\pm$" + str(df['BIS_NADIR'][1]) for df in list_data]
    final_table['BIS_NADIR_min'] = [str(df['BIS_NADIR'][2]) for df in list_data]

    final_table['ST10_mean'] = [str(df['ST10 (min)'][0]) + "$\pm$" + str(df['ST10 (min)'][1]) for df in list_data]
    final_table['ST10_max'] = [str(df['ST10 (min)'][3]) for df in list_data]

    final_table['ST20_mean'] = [str(df['ST20 (min)'][0]) + "$\pm$" + str(df['ST20 (min)'][1]) for df in list_data]
    final_table['ST20_max'] = [str(df['ST20 (min)'][3]) for df in list_data]

    final_table['US_mean'] = [str(df['US'][0]) + "$\pm$" + str(df['US'][1]) for df in list_data]
    final_table['US_max'] = [str(df['US'][3]) for df in list_data]

    styler = final_table.style
    styler.hide(axis='index')
    styler.format(precision=2)
    print(styler.to_latex())
except FileNotFoundError:
    print("No data available to construct table")


# get data
Number_of_patient = 8
phase = 'total'

bool_PID = False
bool_NMPC = False
bool_MMPC = False
bool_MHEMPC = False
try:
    Data_PID = pd.read_csv(folder_path + "result_PID_" + phase + '_n=' + str(Number_of_patient) + '.csv')
    bool_PID = True
except FileNotFoundError:
    print("No data available for PID")

try:
    Data_NMPC = pd.read_csv(folder_path + "result_NMPC_" + phase + '_n=' + str(Number_of_patient) + '.csv')
    bool_NMPC = True
except FileNotFoundError:
    print("No data available for NMPC")

try:
    Data_MMPC = pd.read_csv(folder_path + "result_multi_NMPC_" + phase + '_n=' + str(Number_of_patient) + '.csv')
    bool_MMPC = True
except FileNotFoundError:
    print("No data available for MMPC")

try:
    Data_MHEMPC = pd.read_csv(folder_path + "result_MHE_NMPC_" + phase + '_n=' + str(Number_of_patient) + '.csv')
    bool_MHEMPC = True
except FileNotFoundError:
    print("No data available for MHEMPC")


# find Patient with minimum BIS for each controller
Patient_id_min_PID = 0
Patient_id_min_NMPC = 0
Patient_id_min_MMPC = 0
Patient_id_min_MHEMPC = 0
for patient_id in range(1, Number_of_patient):
    if bool_PID:
        if np.min(Data_PID[str(patient_id)+'_BIS']) < np.min(Data_PID[str(Patient_id_min_PID)+'_BIS']):
            Patient_id_min_PID = patient_id
    if bool_NMPC:
        if np.min(Data_NMPC[str(patient_id)+'_BIS']) < np.min(Data_NMPC[str(Patient_id_min_NMPC)+'_BIS']):
            Patient_id_min_NMPC = patient_id
    if bool_MMPC:
        if np.min(Data_MMPC[str(patient_id)+'_BIS']) < np.min(Data_MMPC[str(Patient_id_min_MMPC)+'_BIS']):
            Patient_id_min_MMPC = patient_id
    if bool_MHEMPC:
        if np.min(Data_MHEMPC[str(patient_id)+'_BIS']) < np.min(Data_MHEMPC[str(Patient_id_min_MHEMPC)+'_BIS']):
            Patient_id_min_MHEMPC = patient_id

print(f"worst patient for MHE-MPC: {Patient_id_min_MHEMPC}")
if bool_PID:
    BIS_PID = Data_PID[str(Patient_id_min_PID)+'_BIS']
if bool_NMPC:
    BIS_NMPC = Data_NMPC[str(Patient_id_min_NMPC)+'_BIS']
if bool_MMPC:
    BIS_MMPC = Data_MMPC[str(Patient_id_min_MMPC)+'_BIS']
if bool_MHEMPC:
    BIS_MHEMPC = Data_MHEMPC[str(Patient_id_min_MHEMPC)+'_BIS']

ts_PID = 1
ts_MPC = 2
if bool_PID:
    Time_PID = np.arange(0, len(BIS_PID)) * ts_PID / 60
if bool_NMPC:
    Time_MPC = np.arange(0, len(BIS_NMPC)) * ts_MPC / 60

if bool_PID:
    Up_PID = Data_PID[str(Patient_id_min_PID)+'_Up']
    Ur_PID = Data_PID[str(Patient_id_min_PID)+'_Ur']
if bool_NMPC:
    Up_NMPC = Data_NMPC[str(Patient_id_min_NMPC)+'_Up']
    Ur_NMPC = Data_NMPC[str(Patient_id_min_NMPC)+'_Ur']
if bool_MMPC:
    Up_MMPC = Data_MMPC[str(Patient_id_min_MMPC)+'_Up']
    Ur_MMPC = Data_MMPC[str(Patient_id_min_MMPC)+'_Ur']
if bool_MHEMPC:
    Up_MHEMPC = Data_MHEMPC[str(Patient_id_min_MHEMPC)+'_Up']
    Ur_MHEMPC = Data_MHEMPC[str(Patient_id_min_MHEMPC)+'_Ur']


# %% Create BIS figure
ax: matplotlib.pyplot.Axes
fig, ax = plt.subplots()
if bool_PID:
    ax.plot(Time_PID, BIS_PID, label='PID')
if bool_NMPC:
    ax.plot(Time_MPC, BIS_NMPC, label='NMPC')
if bool_MMPC:
    ax.plot(Time_MPC, BIS_MMPC, label='MMPC')
if bool_MHEMPC:
    ax.plot(Time_MPC, BIS_MHEMPC, label='MHE_NMPC')
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
savepath = image_folder_path + "worst_bis.pdf"
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()

# Create drug rates figures
linewidth = 1.3
if bool_PID:
    plt.plot(Time_PID, Up_PID, label='Propofol PID (mg/s)',
             linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:blue'])
    plt.plot(Time_PID, Ur_PID, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:blue'],
             label='Remifentanil PID (µg/s)')
if bool_NMPC:
    plt.plot(Time_MPC, Up_NMPC, label='Propofol NMPC (mg/s)',
             linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:orange'])
    plt.plot(Time_MPC, Ur_NMPC, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:orange'],
             label='Remifentanil NMPC (µg/s)')
if bool_MMPC:
    plt.plot(Time_MPC, Up_MMPC, label='Propofol MMPC (mg/s)',
             linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:green'])
    plt.plot(Time_MPC, Ur_MMPC, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:green'],
             label='Remifentanil MMPC (µg/s)')
if bool_MHEMPC:
    plt.plot(Time_MPC, Up_MHEMPC, label='Propofol MHE_NMPC (mg/s)',
             linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:red'])
    plt.plot(Time_MPC, Ur_MHEMPC, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:red'],
             label='Remifentanil MHE_NMPC (µg/s)')

plt.grid(linewidth=0.4)
plt.legend(fontsize=13)
# plt.xlim([-0.2, 10.2])
plt.xlabel('Time (min)', fontsize=13)
plt.draw()


# fig, ax = plt.subplots(2, 1)
# linewidth = 1.3
# ax[0].plot(Time_PID, Up_PID, label='PID', linewidth=linewidth)
# ax[0].plot(Time_MPC, Up_NMPC, label='NMPC', linewidth=linewidth)
# ax[0].plot(Time_MPC, Up_MMPC, label='MMPC', linewidth=linewidth)

# ax[1].plot(Time_PID, Ur_PID, linewidth=linewidth, label='PID')
# ax[1].plot(Time_MPC, Ur_NMPC, linewidth=linewidth, label='NMPC')
# ax[1].plot(Time_MPC, Ur_MMPC, linewidth=linewidth, label='MMPC')

# ax[0].grid(linewidth=0.4)
# ax[1].grid(linewidth=0.4)

# ax[0].legend(fontsize=13)
# ax[1].legend(fontsize=13)

# # ax[0].set_title("Propofol rate")
# # ax[1].set_title("Remifentanil rate")

# ax[0].set_xlim([-0.2, 10.2])
# ax[1].set_xlim([-0.2, 10.2])

# ax[0].set_ylabel('Propofol (mg/s)', fontsize=13)
# ax[1].set_ylabel('Remifentanil (µg/s)', fontsize=13)

# ax[1].set_xlabel('Time (min)', fontsize=13)

# plt.draw()

# save it
savepath = image_folder_path + "worst_bis_inputs" + phase + ".pdf"
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()

# %% Second figure: mean values
transparency = 0.3

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

if bool_MMPC:
    BIS_MMPC = Data_MMPC.loc[:, Data_MMPC.columns.str.contains('BIS')]
    mean_BIS_MMPC = BIS_MMPC.mean(axis=1)
    std_BIS_MMPC = BIS_MMPC.std(axis=1)
    plt.fill_between(Time_PID, mean_BIS_PID-std_BIS_PID, mean_BIS_PID+std_BIS_PID,
                     alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:blue'])

if bool_MHEMPC:
    BIS_MHEMPC = Data_MHEMPC.loc[:, Data_MHEMPC.columns.str.contains('BIS')]
    mean_BIS_MHEMPC = BIS_MHEMPC.mean(axis=1)
    std_BIS_MHEMPC = BIS_MHEMPC.std(axis=1)
    plt.fill_between(Time_MPC, mean_BIS_MHEMPC-std_BIS_MHEMPC, mean_BIS_MHEMPC+std_BIS_MHEMPC,
                     alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:red'])


if bool_PID:
    plt.plot(Time_PID, mean_BIS_PID-std_BIS_PID, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:blue'], linewidth=1)
if bool_NMPC:
    plt.plot(Time_MPC, mean_BIS_NMPC-std_BIS_NMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:orange'], linewidth=1)
if bool_MMPC:
    plt.plot(Time_MPC, mean_BIS_MMPC-std_BIS_MMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:green'], linewidth=1)
if bool_MHEMPC:
    plt.plot(Time_MPC, mean_BIS_MHEMPC-std_BIS_MHEMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:red'], linewidth=1)

if bool_PID:
    plt.plot(Time_PID, mean_BIS_PID+std_BIS_PID, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:blue'], linewidth=1)
if bool_NMPC:
    plt.plot(Time_MPC, mean_BIS_NMPC+std_BIS_NMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:orange'], linewidth=1)
if bool_MMPC:
    plt.plot(Time_MPC, mean_BIS_MMPC+std_BIS_MMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:green'], linewidth=1)
if bool_MHEMPC:
    plt.plot(Time_MPC, mean_BIS_MHEMPC+std_BIS_MHEMPC, linestyle='--',
             color=mcolors.TABLEAU_COLORS['tab:red'], linewidth=1)

if bool_PID:
    plt.plot(Time_PID, mean_BIS_PID, label='PID', color=mcolors.TABLEAU_COLORS['tab:blue'])
if bool_NMPC:
    plt.plot(Time_MPC, mean_BIS_NMPC, label='NMPC', color=mcolors.TABLEAU_COLORS['tab:orange'])
if bool_MMPC:
    plt.plot(Time_MPC, mean_BIS_MMPC, label='MMPC', color=mcolors.TABLEAU_COLORS['tab:green'])
if bool_MHEMPC:
    plt.plot(Time_MPC, mean_BIS_MHEMPC, label='MHE_NMPC', color=mcolors.TABLEAU_COLORS['tab:red'])

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

u = np.linspace(0, 1.5, 100)

J_quadratic = np.square(u)
J_quartic = np.square(J_quadratic)

ue = 0.7
fontsize_plot = 15
fig, ax = plt.subplots()
ax.plot(u, J_quadratic, 'b', label="Quadratic cost")
ax.plot(u, J_quartic, 'r', label="Quartic cost")
ax.axvline(x=ue, ymin=-1, ymax=0.29, linestyle='--', color="black")
ax.annotate('$U_e$', xy=(0.7, 1.5), fontsize=fontsize_plot)
ax.grid(linewidth=0.4)
ax.legend(fontsize=fontsize_plot)
ax.set_xlabel('$U$', fontsize=fontsize_plot)
ax.set_ylabel('$J$', fontsize=fontsize_plot)


# set aspect ratio to 1
ratio = 0.9
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)


plt.draw()

# save it
savepath = image_folder_path + "cost.pdf"
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()
