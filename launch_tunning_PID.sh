#!/bin/bash

#OAR -n tune_PID
#OAR -l /nodes=1/core=18,walltime=01:00:00
#OAR -stderr tune_PID.err
#OAR -stdout tune_PID.out
#OAR --project pr-damon

source .env/bin/activate
python3 scripts/PID_study.py


