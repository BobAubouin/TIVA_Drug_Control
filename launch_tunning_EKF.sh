#!/bin/bash

#OAR -n tune_EKF
#OAR -l /nodes=1/core=32,walltime=04:00:00
#OAR -stderr tune_EKF.err
#OAR -stdout tune_EKF.out
#OAR --project pr-damon

source .env/bin/activate
python3 scripts/EKF_MPC_study.py

