#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:23:55 2023

@author: aubouinb
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, lognorm


# %% theoretical value
np.random.seed(1)


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    """Generate a random float number from a a truncate dnormal distribution."""
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


nb_patient = 15
# mean value
c50p_mean = 4.47
c50r_mean = 19.3
gamma_mean = 1.43
beta_mean = 0
E0_mean = 97.4
Emax_mean = E0_mean
# coefficient of variation
cv_c50p = 0.182
cv_c50r = 0.888
cv_gamma = 0.304
cv_beta = 0
cv_E0 = 0
cv_Emax = 0
# estimation of log normal standard deviation
w_c50p = np.sqrt(np.log(1+cv_c50p**2))
w_c50r = np.sqrt(np.log(1+cv_c50r**2))
w_gamma = np.sqrt(np.log(1+cv_gamma**2))
w_beta = np.sqrt(np.log(1+cv_beta**2))
w_E0 = np.sqrt(np.log(1+cv_E0**2))
w_Emax = np.sqrt(np.log(1+cv_Emax**2))

# Create random values
c50p = c50p_mean*np.exp(np.random.normal(scale=w_c50p, size=nb_patient))
c50r = c50r_mean*np.exp(np.random.normal(scale=w_c50r, size=nb_patient))
gamma = gamma_mean*np.exp(np.random.normal(scale=w_gamma, size=nb_patient))
beta = beta_mean*np.exp(np.random.normal(scale=w_beta, size=nb_patient))
E0 = E0_mean*np.exp(np.random.normal(scale=w_E0, size=nb_patient))
Emax = Emax_mean*np.exp(np.random.normal(scale=w_Emax, size=nb_patient))

age = np.random.randint(low=18, high=70, size=nb_patient)
height = np.random.randint(low=150, high=190, size=nb_patient)
weight = np.random.randint(low=50, high=100, size=nb_patient)
gender = np.random.randint(low=0, high=2, size=nb_patient)


df = pd.DataFrame({'age': age, 'height': height, 'weight': weight, 'gender': gender,
                   'c50p': c50p, 'c50r': c50r, 'gamma': gamma, 'beta': beta, 'E0': E0, 'Emax': Emax})

df.loc[len(df)] = df.mean()
df.loc[len(df)-1, 'gender'] = int(df.loc[len(df)-1, 'gender'])
df.to_csv('./scripts/Patient_table.csv')

df.hist()
