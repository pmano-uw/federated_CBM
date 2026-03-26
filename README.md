# Code for Collaborative and Federated Condition Based Maintenance
This repository contains the code in the paper "Collaborative and Federated Condition Based Maintenance"

## Installation and dependencies
If using `conda`, create the environment with dependencies using
```
conda env create -f environment.yml
```

Otherwise, install dependencies using
```
pip install -r requirements.txt
```

## Replication of the simulation study for parameter estimation
* The results from section 7.1 (parameter estimation) can be replicated by running the notebook `Plot.ipynb` in the folder `linear_degradation_mcmc`.
* The results from section 7.2 (optimal policy), 7.3 (discounted cost), and 7.4 (value function) can be replicated by running the notebook `Plot.ipynb` in the folder `linear_degradation_signal`.
* The results from section 7.5 (real-world case study) can be replicated by running the notebook `Plot.ipynb` in the folder `linear_num_exp` and the notebook `Plot.ipynb` in the folder `linear_num_exp_policy`.
* 