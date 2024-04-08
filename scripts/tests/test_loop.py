import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8


from close_loop_anesth.loop import perform_simulation
from create_param import load_mekf_param, load_mhe_param


NMPC_param = {'N': 60, 'Nu': 60, 'R': 0.1*np.diag([4, 1])}

mekf_param = load_mekf_param([5, 6, 6],
                             q=10,
                             r=1,
                             alpha=10,
                             lambda_2=1e-2,
                             epsilon=0.8)

mhe_param = load_mhe_param(R=0.1, N_mhe=30, vmax=1e4, q=1e3, vmin=0.1)
NMPC_param['bool_non_linear'] = True
age = 27
height = 165
weight = 70
gender = 0

start_time = time.time()


results = perform_simulation([age, height, weight, gender],
                             'induction',
                             'MHE_NMPC',
                             NMPC_param,
                             mhe_param,
                             [True, True],
                             2,
                             bool_noise=False)

print(f"Simulation time: {time.time() - start_time:.2f} s")
# # plot results
plt.subplot(2, 1, 1)
plt.plot(results['Time'], results['BIS'], label='BIS')
plt.plot(results['Time'], results['BIS']*0 + 50, label='BIS target')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(results['Time'], results['u_propo'], label='propo')
plt.plot(results['Time'], results['u_remi'], label='remi')
plt.legend()
plt.grid()

plt.show()
