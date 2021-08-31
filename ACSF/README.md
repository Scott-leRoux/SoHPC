# Atom Centered Symmetry Functions (ACSF) 

This folder contains all files which are used in ACSF models, including Descriptor Creation and Benchmarking, Hyperparmater Tuning with Keras-Tuner and Neural Network Training with 
Benchmarking on GPU and parallel CPUs. 

in order to reproduce code following commands are necessary 

python3 acsf_data.py
python3 NN_acsf.py 

nn_bench.txt and benchmark.txt contain results from benchmarking the Neural Network Training and Feature Engineering (Descriptor Creation) respectively.

out.kt contains the top 10 models found during optimisation

The trained Hyperparmeters were:
	- Learning Rate: {.01, .001, .0001}
	- Activation Fucntion: {softplus, shifted softplus, Exponential Linear Unit (ELU)}
	- Kernel Initialisation: {Xavier Uniform, He Uniform}
	- Batch Size: {32,64,128}
	- Number of Hidden Layers: {1,2,3,4}
	- Hidden Layer Size: {32, 64, 128, 256}


