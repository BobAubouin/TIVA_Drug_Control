import matplotlib.pyplot as plt
import numpy as np
import time
import os


from close_loop_anesth.loop import perform_simulation
from create_param import load_mekf_param, load_mhe_param


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


NMPC_param = {'N': 60, 'Nu': 60, 'R': 0.1*np.diag([4, 1])}

mekf_param = load_mekf_param([5, 6, 6],
                             q=10,
                             r=1,
                             alpha=10,
                             lambda_2=1e-2,
                             epsilon=0.8)

mhe_param = load_mhe_param(R=0.1, N_mhe=30, vmax=1e4, q=1e3, vmin=0.1)

age = 27
height = 165
weight = 70
gender = 0

start_time = time.time()

with suppress_stdout_stderr():
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
