import matplotlib.pyplot as plt
import numpy as np

vmax = 1e4
vmin = 0.1
theta = [vmin, vmax, 100, 0.02]*4
theta[12] = vmax
theta[13] = -vmax+vmin


def exp_theta(theta, time):
    return theta[0] + theta[1]*np.exp(-theta[2]*np.exp(-theta[3]*time))


time = np.linspace(0, 20*60, 1000)
plt.plot(time, exp_theta(theta[:4], time), label='Q1')
plt.plot(time, exp_theta(theta[4:8], time), label='Q2')
plt.plot(time, exp_theta(theta[8:12], time), label='Q3')
plt.plot(time, exp_theta(theta[12:16], time), label='Q4')
plt.legend()
plt.grid()
plt.yscale('log')
plt.show()
