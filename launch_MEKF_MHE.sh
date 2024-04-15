#!/bin/bash

#OAR -n tune_MEKF_MHE_MPC
#OAR -l /nodes=1/core=32,walltime=03:00:00
#OAR -stderr tune_MEKF_MHE_MPC.err
#OAR -stdout tune_MEKF_MHE_MPC.out
#OAR --project pr-damon

source .venv/bin/activate
python3 scripts/studies/study_mekf_mhe.py


