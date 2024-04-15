import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8


from close_loop_anesth.experiments import auto_tune_pid, random_simu

caseid = 56
control_type = 'PID'
phase = 'total'
ratio = 2

start_time = time.time()
control_param = auto_tune_pid(caseid, phase, 'IAE_biased_normal', ratio, nb_of_step=1000)

control_param['ratio'] = ratio

results = random_simu(caseid, phase, control_type, control_param, output='dataframe')

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
