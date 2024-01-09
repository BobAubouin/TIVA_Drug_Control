from matplotlib import rc
import matplotlib.pyplot as plt
from MEKF_MHE_MPC_study import small_obj
from MEKF_MHE_MPC_study import MHE_param, parem_mekf_mhe, training_patient
import numpy as np
from python_anesthesia_simulator import metrics

N = 30
R = 42 * np.diag([4, 1])
t_switch = 120
phase = 'induction'

MPC_param = [N, N, R]
parem_mekf_mhe[-1] = t_switch
mhe_nmpc_param = parem_mekf_mhe + MPC_param

df_list = []
TT_list = []
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
    print(f"Patient {counter}/{len(training_patient)} done")


print(f"Training TT : {np.mean(TT_list)}")

# plot all BIS
rc('text', usetex=True)
rc('font', family='serif')
plt.figure(figsize=(8, 4))
for i in range(len(df_list)):
    plt.plot(df_list[i]['Time'], df_list[i]['BIS'], label=f"Patient {i}")
plt.xlabel("Time (min)")
plt.ylabel("BIS")
# plt.legend()
plt.grid()
plt.show()
