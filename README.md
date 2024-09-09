# TIVA_Drug_Control

Git depository associated to the paper: B. Aubouin–Pairault, M. Fiacchini, and T. Dang, ***“Online identification of pharmacodynamic parameters for closed-loop anesthesia with model predictive control,”*** Computers & Chemical Engineering, vol. 191, p. 108837, Dec. 2024, doi: 10.1016/j.compchemeng.2024.108837. Please cite us if you use this code.

Use the [Python Anesthesia Simulator](https://github.com/BobAubouin/Python_Anesthesia_Simulator) to run the simulation and compute the performances metrics.

**Abstract:**
In this paper, a controller is proposed to automate the injection of propofol and remifentanil during general anesthesia using bispectral index (BIS) measurement. To handle the parameter uncertainties due to inter- and intra-patient variability, an extended estimator is used coupled with a Model Predictive Controller (MPC). Two methods are considered for the estimator: the first one is a multiple extended Kalman filter (MEKF), and the second is a moving horizon estimator (MHE). The state and parameter estimations are then used in the MPC to compute the next drug rates. The methods are compared with a PID from the literature. The robustness of the controller is evaluated using Monte-Carlo simulations on a wide population, introducing uncertainties in all parts of the model. Results both on the induction and maintenance phases of anesthesia show the potential interest in using this adaptive method to handle parameter uncertainties.


## Installation

Install all the required packages with the command:

```
pip install .
```

## Usage

## Reproduce the results
Launch each study located in the scripts/studies folder. 

**Caution:** The simulation scripts may take time to run (approx 5h on a 32 thread server for the MPC ones).

The results can be found in the plot results notebook.

## Use the package

The package can be used to simulate the anesthesia induction phase with the MPC and PID controllers. 
A simple example is given in the test notebook, for more in depth change the files of the package can be modified.

## Authors

Bob Aubouin--Pairault, Mirko Fiacchini, Thao Dang

## License

 GPL-3.0
