import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, activations
from tensorflow.keras import initializers 
from tensorflow.keras import optimizers
import numpy as np
import keras_tuner as kt

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
physical_devices = tf.config.list_physical_devices('GPU')

D_train = np.load("acsf_f_train.npy")
D_test = np.load("acsf_f_test.npy")
D_val = np.load("acsf_f_val.npy")

E_test = np.load("energy_f_test.npy")
E_val = np.load("energy_f_val.npy")
E_train = np.load("energy_f_train.npy")


n_features = D_train.shape[1]
n_energy_output = 1 
outputs = []

def ssp(x):
	return tf.math.softplus(x) + tf.math.log(.5)

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1 , patience = 40)
callbacks = [es]

def model_builder(hp):
	model = Sequential()
	model.add(tf.keras.Input(n_features))

	hp_ker_init = hp.Choice('kernel_initializer', ['he_uniform', 'glorot_uniform'])
	hp_act_fn = hp.Choice('activation', values = ['softplus', 'ssp','elu'])

	if hp_act_fn == 'softplus':
		model.add(Dense(hp.Choice('layer_0_units', values = [64,128,256]), activation = hp_act_fn, 
					kernel_initializer = hp_ker_init))
	elif hp_act_fn == 'ssp':
		model.add(Dense(hp.Choice('layer_0_units', [64,128,256]), kernel_initializer = hp_ker_init))
		model.add(layers.Activation(ssp))
	else:
		model.add(Dense(hp.Choice('layer_0_units', [64,128,256]), kernel_initializer = hp_ker_init))
		model.add(layers.ELU())
	
	hp_num_hidden = hp.Int('units', min_value = 0, max_value = 4)
	for i in range(hp_num_hidden):
		j = i + 1
		model.add(Dense(hp.Choice(f'layer_{j}_units', values = [16,32, 64, 128]), kernel_initializer = hp_ker_init))
		if hp_act_fn == 'softplus':
			model.add(layers.Activation('softplus'))
		elif hp_act_fn == 'ssp':
			model.add(layers.Activation(ssp))
		else:
			model.add(layers.ELU())

	model.add(Dense(1))
	opt = optimizers.Adam(learning_rate = hp.Choice('learning_rate', [.001, .0001]))
	model.compile(optimizer = opt, loss = 'mse')
	return model 

					
class CustomTuner(kt.tuners.BayesianOptimization):
	def run_trial(self, trial, *args, **kwargs):
		kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size' , [32,64])
		super(CustomTuner, self).run_trial(trial, *args, **kwargs)

myTuner = CustomTuner(model_builder,
                        objective = 'val_loss',
                        executions_per_trial = 1,
                        max_trials = 30,
                        directory = 'kt_results',
                        project_name = "Custom_tuner_bayes",
                        overwrite = True,
			)



myTuner.search(D_train, E_train, callbacks = callbacks, validation_data = (D_val, E_val), epochs = 120, verbose = 1)
best_hps=myTuner.get_best_hyperparameters()[0].values
print(best_hps)
print(myTuner.results_summary())
print(myTuner.get_best_models()[0].summary())	
	
	
