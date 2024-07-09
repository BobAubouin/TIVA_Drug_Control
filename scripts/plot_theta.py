import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# plot config
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font = {'family': 'serif',
        'weight': 'normal',
        'size': 16}


plt.rc('text', usetex=True)
matplotlib.rc('font', **font)

vmax = 1e4
vmin = 0.01
theta = [vmin, vmax, 100, 0.02]*4
theta[12] = vmax
theta[13] = -vmax+vmin


def exp_theta(theta, time):
    return theta[0] + theta[1]*np.exp(-theta[2]*np.exp(-theta[3]*time))


time = np.linspace(0, 20*60, 1000)
plt.figure(figsize=(5, 3.5))
plt.plot(time/60, exp_theta(theta[:4], time), label='PD parameters weight')
# plt.plot(time/60, exp_theta(theta[4:8], time), label='Q2')
# plt.plot(time/60, exp_theta(theta[8:12], time), label='Q3')
plt.plot(time/60, exp_theta(theta[12:16], time), label='Disturbance weight')
plt.legend()
plt.grid()
plt.xlabel('Time (min)')
# plt.ylabel('Value')
plt.yscale('log')
plt.tight_layout()
plt.savefig('./outputs/theta.pdf')
plt.show()
