#!/bin/bash

#OAR -n tune_PID
#OAR -l /nodes=1/core=32,walltime=00:10:00
#OAR -stderr tune_PID.err
#OAR -stdout tune_PID.out
#OAR --project pr-damon

source .venv/bin/activate
python3 scripts/studies/study_pid.py


