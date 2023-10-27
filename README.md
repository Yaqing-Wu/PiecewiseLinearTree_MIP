# Piecewise Linear Trees as Surrogate Models for System Design and Planning under High-frequency Temporal Variability

This repository contains code for the numerical experiments in subsection 4.1 (N-dimensional quadratic functions) and subsection 4.2 (Real-world or synthetic datasets). 

## Repository content
- **quadratic (N-dimensional quadratic functions)**
	- quadratic_model_training.ipynb: train the machine learning models and save the trained models
	- quadratic_opt_pwl_tree.ipynb: build and solve the MIP model corresponding to the trained PWL tree models
	- quadratic_opt_pwc_tree.ipynb: build and solve the MIP model corresponding to the trained CARTs
	- quadratic_opt_relu.ipynb: build and solve the MIP model corresponding to the trained ReLU networks
	- quadratic_opt_rf.ipynb: build and solve the MIP model corresponding to the trained random forests
	- helpers: a folder containing helper functions
		- quadratic.py: contains the function that generates the quadratic dataset
		- pwl_tree.py: contains helper functions for the MIP model corresponding to the trained PWL tree models
		- rf.py: contains helper functions for the MIP model corresponding to the trained random forests
- **real_synthetic_data (Real-world or synthetic datasets)**
	- PWL tree
		- Powerplant_pwl.ipynb: train PWL trees and solve the MIP model corresponding to the trained PWL tree models for the power plant dataset
		- Friedman1_max_pwl.ipynb: train PWL trees and solve the MIP model corresponding to the trained PWL tree models for the Friedman #1 dataset maximization problem
		- Friedman1_min_pwl.ipynb: solve the MIP model corresponding to the trained PWL tree models obtained from Friedman1_max_pwl.ipynb for the Friedman #1 dataset minimization problem
		- Friedman3_max_pwl.ipynb: train PWL trees and solve the MIP model corresponding to the trained PWL tree models for the Friedman #3 dataset maximization problem
		- Friedman3_min_pwl.ipynb: solve the MIP model corresponding to the trained PWL tree models obtained from Friedman3_max_pwl.ipynb the Friedman #3 dataset minimization problem
	- Benchmark methods (use the code for the power plant dataset as the examples, and the code for the other datasets could be obtained by replacing the datasets used in the code)
		- Powerplant_pwc.ipynb: for CARTS
		- Powerplant_relu.ipynb: for ReLU networks
		- Powerplant_rf.ipynb: for random forests
	- helpers: a folder containing helper functions (note that the helper functions are the same as the one for quadratic function numerical experiments, and we have a copy here to enable correct package importing in the code)
		- pwl_tree.py: contains helper functions for the MIP model corresponding to the trained PWL tree models
		- rf.py: contains helper functions for the MIP model corresponding to the trained random forests
	- power_plant_csv: the csv file for the power plant data from UCI

## Packages used
- gurobi                    9.1.2
- jupyterlab                3.4.4
- mlinsights                0.3.543
- numpy                     1.21.5
- pandas                    1.3.5
- scikit-learn              0.24.2
- tqdm                      4.64.0

## Citation
If ou use this work, please cite the following paper:
::
@article{WU2023,
title = {Piecewise Linear Trees as Surrogate Models for System Design and Planning under High-frequency Temporal Variability},
journal = {European Journal of Operational Research},
year = {2023},
issn = {0377-2217},
doi = {https://doi.org/10.1016/j.ejor.2023.10.028},
url = {https://www.sciencedirect.com/science/article/pii/S0377221723008032},
author = {Yaqing Wu and Christos T. Maravelias}
}

## References
- The power plant dataset comes from:
	Tfekci,Pnar and Kaya,Heysem. (2014). Combined Cycle Power Plant. UCI Machine Learning Repository. https://doi.org/10.24432/C5002N.
- Part of the helper functions for random forests (i.e., functions in rf.py) come from:
	Lin, Yifan. (2018). https://github.com/ivanlin9522/TREE
