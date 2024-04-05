import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8

from close_loop_anesth.experiments import random_simu, training_patient, compute_cost
from create_param import load_mhe_param

NMPC_param = {'N': 65, 'Nu': 65, 'R': 0.05*np.diag([4, 1])}

mhe_param = load_mhe_param(R=0.001, N_mhe=25, vmax=1e4, q=1e3, vmin=0.01)

start_time = time.time()

# for case in training_patient:
#     df = random_simu(caseid=case,
#                      control_type='MHE_NMPC',
#                      control_param=NMPC_param,
#                      estim_param=mhe_param,
#                      output='dataframe',
#                      phase='induction')

#     cost = compute_cost(df, 'IAE_biased')
#     print(f"case: {case}, Cost: {cost}")

df = random_simu(caseid=47,
                 control_type='MHE_NMPC',
                 control_param=NMPC_param,
                 estim_param=mhe_param,
                 output='dataframe',
                 phase='induction')


print(f"Simulation time: {time.time() - start_time:.2f} s")

cost = compute_cost(df, 'IAE_biased')
print(f"Cost: {cost}")

# plot results
plt.subplot(2, 1, 1)
plt.plot(df['Time'], df['BIS'], label='BIS')
plt.plot(df['Time'], df['BIS']*0 + 50, label='BIS target')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(df['Time'], df['u_propo'], label='propo')
plt.plot(df['Time'], df['u_remi'], label='remi')
plt.legend()
plt.grid()

plt.show()
