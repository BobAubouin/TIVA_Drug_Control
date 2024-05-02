# TIVA_Drug_Control

Git depository associated to the paper "PID and Model Predictive Control Approach for Drug Dosage in Anesthesia During Induction: a Comparative Study".

Use the [Python Anesthesia Simulator](https://github.com/BobAubouin/Python_Anesthesia_Simulator) to run the simulation and compute the performances metrics.

**Abstract:**
In this paper, a PID controller is compared to an extended moving horizon estimator coupled with a model predictive control approach for the problem of dosing propofol and remifentanil during the induction of anesthesia. For the PID controller, taken from the literature, a fixed ratio is considered between propofol and remifentanil flow rates and an anti-windup strategy is used to prevent integration wind-up. The optimal control approach uses an MHE to estimate both the states and the pharmacodynamic parameters of the system, followed by a non-linear model predictive controller to compute the optimal drug rates according to the model. Both controllers are tuned using the same criterion, and are compared by simulating 500 uncertain patient models for the induction phase. 


# Installation

Install all the required packages with the command:

```
pip install .
```

## Usage

# Reproduce the results
Launch each study located in the scripts/studies folder. 

**Caution:** The simulation scripts may take time to run (approx 5h on a 32 thread server for the MPC ones).

The results can be found in the plot results notebook.

# Use the package

The package can be used to simulate the anesthesia induction phase with the MPC and PID controllers. 
A simple example is given in the test notebook, for more in depth change the files of the package can be modified.

## Authors

Bob Aubouin--Pairault, Mirko Fiacchini, Thao Dang

## License

 GPL-3.0
