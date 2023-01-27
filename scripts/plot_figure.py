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

# get data
Number_of_patient = 500
phase = 'induction'


Patient_id = 108


Data_PID = pd.read_csv("./Results_data/result_PID_n=" + str(Number_of_patient) + '.csv')
Data_MMPC = pd.read_csv("./Results_data/result_multi_NMPC_n=" + str(Number_of_patient) + '.csv')
Data_NMPC = pd.read_csv("./Results_data/result_NMPC_n=" + str(Number_of_patient) + '.csv')

BIS_PID = Data_PID[str(Patient_id)+'_BIS']
BIS_NMPC = Data_NMPC[str(Patient_id)+'_BIS']
BIS_MMPC = Data_MMPC[str(Patient_id)+'_BIS']

ts_PID = 1
ts_MPC = 2
Time_PID = np.arange(0, len(BIS_PID)) * ts_PID / 60
Time_MPC = np.arange(0, len(BIS_NMPC)) * ts_MPC / 60

Up_PID = Data_PID[str(Patient_id)+'_Up']
Ur_PID = Data_PID[str(Patient_id)+'_Ur']
Up_NMPC = Data_NMPC[str(Patient_id)+'_Up']
Ur_NMPC = Data_NMPC[str(Patient_id)+'_Ur']
Up_MMPC = Data_MMPC[str(Patient_id)+'_Up']
Ur_MMPC = Data_MMPC[str(Patient_id)+'_Ur']


# %% Create BIS figure
ax: matplotlib.pyplot.Axes
fig, ax = plt.subplots()
ax.plot(Time_PID, BIS_PID, label='PID')
ax.plot(Time_MPC, BIS_NMPC, label='NMPC')
ax.plot(Time_MPC, BIS_MMPC, label='MMPC')
ax.grid(linewidth=0.4)
ax.legend(fontsize=13)

ax.set_yticks(list(ax.get_yticks()) + [int(BIS_PID[0])])
ax.set_ylim([23, 97])
ax.set_xlim([-0.2, 10.2])
ax.set_xlabel('Time (min)', fontsize=13)
ax.set_ylabel('BIS', fontsize=13)
ygridlines = ax.get_ygridlines()
gridline_of_interest = ygridlines[-1]
gridline_of_interest.set_visible(False)
plt.draw()

# save it
savepath = "./Results_Images/BIS_case="+str(Patient_id) + ".pdf"
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()

# Create drug rates figures
linewidth = 1.3
plt.plot(Time_PID, Up_PID, label='Propofol PID (mg/s)', linewidth=linewidth)
plt.plot(Time_MPC, Up_NMPC, label='Propofol NMPC (mg/s)', linewidth=linewidth)
plt.plot(Time_MPC, Up_MMPC, label='Propofol MMPC (mg/s)', linewidth=linewidth)

plt.plot(Time_PID, Ur_PID, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:blue'],
         label='Remifentanil PID (µg/s)')
plt.plot(Time_MPC, Ur_NMPC, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:orange'],
         label='Remifentanil NMPC (µg/s)')
plt.plot(Time_MPC, Ur_MMPC, linestyle=(0, (3, 1)), linewidth=linewidth, color=mcolors.TABLEAU_COLORS['tab:green'],
         label='Remifentanil MMPC (µg/s)')

plt.grid(linewidth=0.4)
plt.legend(fontsize=13)
plt.xlim([-0.2, 10.2])
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
savepath = "./Results_Images/Rates_case="+str(Patient_id) + ".pdf"
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()

# %% Second figure: mean values

BIS_PID = Data_PID.loc[:, Data_PID.columns.str.contains('BIS')]
mean_BIS_PID = BIS_PID.mean(axis=1)
std_BIS_PID = BIS_PID.std(axis=1)

BIS_NMPC = Data_NMPC.loc[:, Data_NMPC.columns.str.contains('BIS')]
mean_BIS_NMPC = BIS_NMPC.mean(axis=1)
std_BIS_NMPC = BIS_NMPC.std(axis=1)

BIS_MMPC = Data_MMPC.loc[:, Data_MMPC.columns.str.contains('BIS')]
mean_BIS_MMPC = BIS_MMPC.mean(axis=1)
std_BIS_MMPC = BIS_MMPC.std(axis=1)

transparency = 0.3
plt.fill_between(Time_MPC, mean_BIS_MMPC-std_BIS_MMPC, mean_BIS_MMPC+std_BIS_MMPC,
                 alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:green'])  # , hatch="\\\\")

plt.fill_between(Time_MPC, mean_BIS_NMPC-std_BIS_NMPC, mean_BIS_NMPC+std_BIS_NMPC,
                 alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:orange'])

plt.fill_between(Time_PID, mean_BIS_PID-std_BIS_PID, mean_BIS_PID+std_BIS_PID,
                 alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:blue'])  # , hatch="////")

plt.plot(Time_PID, mean_BIS_PID, label='PID', color=mcolors.TABLEAU_COLORS['tab:blue'])
plt.plot(Time_MPC, mean_BIS_NMPC, label='NMPC', color=mcolors.TABLEAU_COLORS['tab:orange'])
plt.plot(Time_MPC, mean_BIS_MMPC, label='MMPC', color=mcolors.TABLEAU_COLORS['tab:green'])

plt.grid(linewidth=0.4)
plt.legend(fontsize=13)
plt.xlabel('Time (min)', fontsize=13)
plt.ylabel('BIS', fontsize=13)
plt.ylim([40, 100])
plt.xlim([-0.2, 10.2])
plt.draw()

# save it
savepath = "./Results_Images/BIS_mean_case.pdf"
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
savepath = "./Results_Images/cost.pdf"
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()
