# TIVA_Drug_Control

Git depository associated to the paper "Automated Multi-Drugs Administration During Total Intravenous Anesthesia Using Multi-Model Predictive Control".

Use the [Python Anesthesia Simulator](https://github.com/BobAubouin/Python_Anesthesia_Simulator) to run the simulation and compute the performances metrics.

**Abstract:**
In this paper, a multi-model predictive control approach is used to automate the co-administration of Propofol and Remifentanil from BIS measurement during general anesthesia. To handle the parameter uncertainties in the non-linear output function, multiple Extended Kalman Filters are used to estimate the state of the system in parallel. The best model is chosen using a model-matching criterion and used in a non-linear MPC to compute the next drug rates. The method is compared with a conventional non-linear MPC approach and a PID from the literature. The robustness of the controller is tested using Monte-Carlo simulations on a wide population introducing uncertainties in the models. Both simulation setup and controller codes are accessible in open source for further use. Our preliminary results show an interest in a multi-model method to handle parameter uncertainties.


# Installation 

Install all the required packages with the command:

```python
pip install - r requirement.txt
```

## Usage

Launch tunning and simulation scripts in the dedicated folder. To visualize the result of the simulation script run *read_simulation_results.py*.

**Caution:** The simulation scripts may take time to run (approx 1h30 on my PC for the MMPC).

## Authors

Bob Aubouin--Pairault, Mirko Fiacchini, Thao Dang

## License

 GPL-3.0
