# Input Pipeline

This folder contains an alternative version of training a Neural Network on the Coulomb Matrix descriptor of 841 features. 
The alternative version includes use of the tf.data API which improves speed up times of Neural Network training. 

methods of the API used:

	- convert the numpy data to tensors, and create a tf.data.Dataset using tf.data.Dataset.from_tensor_slices()
	
	- call repeat(), shuffle(), to repeat the dataset for infinite epochs and shuffle each iteration. 
	
	- Batch(batch_size) the dataset with the correct batch_size to invoke mini batching. 
	
	- Finally call prefetch() to get the CPU to fetch the next batch whilst the CPU is training. 

The total speed-up is seen in the out.input_pipeline file where we see a greater than 20 second speed up over 100 epochs which is a speed up of ~.2 second speed up per epoch.  

we ran the code in Graph Computation mode to optimise performance of tf.data API methods. 

