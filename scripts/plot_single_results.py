import json

import pandas as pd
import matplotlib.pyplot as plt


# study to load
study_name = 'PID_1'

# load the study
with open(f'data/logs/{study_name}.json', 'r') as f:
    dict = json.load(f)

# print the parameters
print(dict)

# load the results
filename = f'data/signals/{dict["filename"]}.csv'
results = pd.read_csv(filename)

# plot the results
plt.subplot(2, 1, 1)
mean_bis = results.groupby('Time')['BIS'].mean()
std_bis = results.groupby('Time')['BIS'].std()

plt.plot(mean_bis.index, mean_bis, label='BIS')
plt.fill_between(mean_bis.index, mean_bis - std_bis, mean_bis + std_bis, alpha=0.5)
plt.plot(mean_bis.index, mean_bis*0 + 50, label='BIS target')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
mean_propo = results.groupby('Time')['u_propo'].mean()
std_propo = results.groupby('Time')['u_propo'].std()
mean_remi = results.groupby('Time')['u_remi'].mean()
std_remi = results.groupby('Time')['u_remi'].std()

plt.plot(mean_propo.index, mean_propo, label='propo')
plt.fill_between(mean_propo.index, mean_propo - std_propo, mean_propo + std_propo, alpha=0.5)
plt.plot(mean_remi.index, mean_remi, label='remi')
plt.fill_between(mean_remi.index, mean_remi - std_remi, mean_remi + std_remi, alpha=0.5)
plt.legend()
plt.grid()

plt.show()
