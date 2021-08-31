# MBTR (Many Body Tensor Representation)

This folder contains code used for working with the MBTR molecular descriptor. If using to replicate it is necessary to run commands

- python3 mbtr_data_qm9.py <data>
- python3 mbtr_tune.py 

The MBTR models were trained across the following parameters:
	- Learning Rate
	- Network Topology 
	- Activation Function 
	- Kernel Initializer

out.kt contains the logs from tuning MBTR models which describes the top 10 most effective models
