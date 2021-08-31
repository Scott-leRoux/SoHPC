# Force Field Models 

This folder contains work on creating models which calculate both Energy and Forces of molecules which aren't in equilibrium. 

We take the formula that defines $F = \frac{dE}{dr}$ where $r$ is the atomic positions and E = potential Energy, this means we can calculate the Energy with a single Neuron output Layer Neural Networkmodel and then use the above formula to calculate the forces then use both force prediction and Energy prediction in the loss function in order to train the Neural Network. 

Using Dscribe 1.1 the SOAP descriptor has derivatives implementation which is what is needed for the function.

As the model is twice as deep as a usual model it is necessary to use an activation function with continuous second derivative, i.e sigmoid, softplus etc. (smooth activation functions).

To undertake training we need to use tf.GradientTape() API, which allows for custom training loops with our custom loss function. We need a Nested Gradient Tape in order to calculate derivatives of output w.r.t input => loss value w.r.t trainable variables. In Eager Execution this is very slow, hence we use the decorator @tf.function to speed up which is approx 10x. It is important to first debug in Eager Mode.

File information: 

	- ForceFieldML.py : Force Field Model for simple "HH" molecule, using 200 samples and the LennardJones calculator. 
	- tune_ffnet : File 
