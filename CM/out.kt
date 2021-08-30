Best val_loss So Far: 0.004094602074474096
Total elapsed time: 07h 19m 41s
{'activation': 'softplus', 'layer_0_units': 256, 'units': 0, 'learning_rate': 0.0001, 'layer_1_units': 16, 'layer_2_units': 32, 'layer_3_units': 64, 'layer_4_units': 128}
Results summary
Results in kt_results/Random_Search
Showing 10 best trials
Objective(name='val_loss', direction='min')
Trial summary
Hyperparameters:
activation: softplus
layer_0_units: 256
units: 0
learning_rate: 0.0001
layer_1_units: 16
layer_2_units: 32
layer_3_units: 64
layer_4_units: 128
Score: 0.004094602074474096
Trial summary
Hyperparameters:
activation: softplus
layer_0_units: 128
units: 1
learning_rate: 0.0001
layer_1_units: 16
layer_2_units: 128
layer_3_units: 16
layer_4_units: 64
Score: 0.004179293755441904
Trial summary
Hyperparameters:
activation: softplus
layer_0_units: 256
units: 3
learning_rate: 0.0001
layer_1_units: 16
layer_2_units: 16
layer_3_units: 16
Score: 0.00489829620346427
Trial summary
Hyperparameters:
activation: softplus
layer_0_units: 128
units: 2
learning_rate: 0.0001
layer_1_units: 32
layer_2_units: 16
layer_3_units: 32
layer_4_units: 32
Score: 0.005013528745621443
Trial summary
Hyperparameters:
activation: elu
layer_0_units: 256
units: 1
learning_rate: 0.0001
layer_1_units: 16
layer_2_units: 16
layer_3_units: 128
layer_4_units: 64
Score: 0.0053921774961054325
Trial summary
Hyperparameters:
activation: softplus
layer_0_units: 512
units: 2
learning_rate: 0.0001
layer_1_units: 16
layer_2_units: 32
layer_3_units: 64
layer_4_units: 16
Score: 0.005941204726696014
Trial summary
Hyperparameters:
activation: softplus
layer_0_units: 256
units: 2
learning_rate: 0.0001
layer_1_units: 128
layer_2_units: 32
layer_3_units: 32
layer_4_units: 16
Score: 0.006054590456187725
Trial summary
Hyperparameters:
activation: softplus
layer_0_units: 512
units: 3
learning_rate: 0.0001
layer_1_units: 128
layer_2_units: 32
layer_3_units: 32
layer_4_units: 32
Score: 0.006324189715087414
Trial summary
Hyperparameters:
activation: ssp
layer_0_units: 512
units: 3
learning_rate: 0.0001
layer_1_units: 128
layer_2_units: 128
layer_3_units: 16
layer_4_units: 64
Score: 0.0063399141654372215
Trial summary
Hyperparameters:
activation: softplus
layer_0_units: 512
units: 2
learning_rate: 0.0001
layer_1_units: 128
layer_2_units: 128
layer_3_units: 128
layer_4_units: 128
Score: 0.006349343340843916
None
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 256)               215552    
_________________________________________________________________
activation (Activation)      (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 257       
=================================================================
Total params: 215,809
Trainable params: 215,809
Non-trainable params: 0
_________________________________________________________________

