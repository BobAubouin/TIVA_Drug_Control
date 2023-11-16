#!/bin/bash

#OAR -n tune_MPC
#OAR -l /nodes=1/core=32,walltime=04:00:00
#OAR -stderr tune_MPC.err
#OAR -stdout tune_MPC.out
#OAR --project pr-damon

source .env/bin/activate
python3 scripts/MEKF_MPC_study.py

