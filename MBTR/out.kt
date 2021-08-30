
Best val_loss So Far: 0.38646605610847473
Total elapsed time: 12h 55m 44s
{'activation': 'elu', 'layer_0_units': 64, 'units': 3, 'learning_rate': 0.001, 'layer_1_units': 64, 'layer_2_units': 32, 'layer_3_units': 128, 'layer_4_units': 32}
Results summary
Results in kt_results/Random_Search_mbtr
Showing 10 best trials
Objective(name='val_loss', direction='min')
Trial summary
Hyperparameters:
activation: elu
layer_0_units: 64
units: 3
learning_rate: 0.001
layer_1_units: 64
layer_2_units: 32
layer_3_units: 128
layer_4_units: 32
Score: 0.38646605610847473
Trial summary
Hyperparameters:
activation: elu
layer_0_units: 128
units: 3
learning_rate: 0.001
layer_1_units: 128
layer_2_units: 32
layer_3_units: 16
layer_4_units: 32
Score: 0.3910168409347534
Trial summary
Hyperparameters:
activation: ssp
layer_0_units: 128
units: 3
learning_rate: 0.001
layer_1_units: 32
layer_2_units: 128
layer_3_units: 16
layer_4_units: 16
Score: 0.4780529737472534
Trial summary
Hyperparameters:
activation: ssp
layer_0_units: 256
units: 4
learning_rate: 0.001
layer_1_units: 64
layer_2_units: 32
layer_3_units: 128
layer_4_units: 128
Score: 0.5212278962135315
Trial summary
Hyperparameters:
activation: ssp
layer_0_units: 256
units: 2
learning_rate: 0.001
layer_1_units: 32
layer_2_units: 32
layer_3_units: 128
layer_4_units: 16
Score: 0.6403756141662598
Trial summary
Hyperparameters:
activation: ssp
layer_0_units: 128
units: 2
learning_rate: 0.001
layer_1_units: 64
layer_2_units: 32
layer_3_units: 128
layer_4_units: 128
Score: 0.8233163952827454
Trial summary
Hyperparameters:
activation: elu
layer_0_units: 256
units: 1
learning_rate: 0.001
layer_1_units: 64
layer_2_units: 64
layer_3_units: 32
layer_4_units: 16
Score: 0.8256919384002686
Trial summary
Hyperparameters:
activation: ssp
layer_0_units: 64
units: 1
learning_rate: 0.001
layer_1_units: 128
layer_2_units: 32
layer_3_units: 32
layer_4_units: 64
Score: 0.8734954595565796
Trial summary
Hyperparameters:
activation: elu
layer_0_units: 128
units: 3
learning_rate: 0.0001
layer_1_units: 64
layer_2_units: 128
layer_3_units: 16
layer_4_units: 64
Score: 0.8909549117088318
Trial summary
Hyperparameters:
activation: softplus
layer_0_units: 64
units: 4
learning_rate: 0.001
layer_1_units: 64
layer_2_units: 32
layer_3_units: 128
layer_4_units: 128
Score: 0.8913766741752625
None
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                608064    
_________________________________________________________________
elu (ELU)                    (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160      
_________________________________________________________________
elu_1 (ELU)                  (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 32)                2080      
_________________________________________________________________
elu_2 (ELU)                  (None, 32)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 128)               4224      
_________________________________________________________________
elu_3 (ELU)                  (None, 128)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 129       
=================================================================
Total params: 618,657
Trainable params: 618,657
Non-trainable params: 0
_


