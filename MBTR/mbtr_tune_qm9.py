import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, activations
from tensorflow.keras import initializers 
from tensorflow.keras import optimizers
import numpy as np
import keras_tuner as kt
import math
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

D_train, E_train, Z_train = [], [], []

for i in range(1,5):
	D_train.append(np.load(f"mbtr_qm9_train{i}.npy"))
	E_train.append(np.load(f"energies_qm9_train{i}.npy"))
	E_train[i-1] = np.reshape(E_train[i-1], (-1,1))
	Z_train.append(np.load(f"charges_qm9_train{i}.npy"))
D_train_stck = np.vstack(D_train)
E_train_stck = np.vstack(E_train)
Z_train_stck = np.vstack(Z_train)

D_test = np.load("mbtr_qm9_test.npy")
D_val = np.load("mbtr_qm9_train0.npy")

E_test = np.load("energies_qm9_test.npy")
E_val = np.load("energies_qm9_train0.npy")

Z_test = np.load("charges_qm9_test.npy")
Z_val = np.load("charges_qm9_train0.npy")

#normalize data 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(D_train_stck)
D_train_stck_n = scaler.fit_transform(D_train_stck)
D_val_n = scaler.fit_transform(D_val)
n_features = D_train_stck.shape[1]
n_energy_output = 1 
n_charges_output = Z_train_stck.shape[1]
outputs = []

print(D_test.shape)


es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1 , patience = 30)
mc = ModelCheckpoint("best_model", monitor = 'val_loss', mode = 'min', save_best_only = True, verbose = 1)

callbacks = [es]

def shifted_softplus(x):
	return tf.math.softplus(x) + tf.math.log(.5)



def model_builder(hp):
	model = Sequential()
	model.add(tf.keras.Input(n_features))
	hp_act_fn = hp.Choice('activation', values = ['ssp', 'softplus', 'elu'])
	if hp_act_fn == 'softplus':
		model.add(Dense(hp.Choice(f'layer_{0}_units', values = [256,128,64])))
		model.add(layers.Activation('softplus'))
	elif hp_act_fn == 'ssp':
		model.add(Dense(hp.Choice(f'layer_{0}_units', values = [256, 64, 128]), activation = shifted_softplus))
	else:
		model.add(Dense(hp.Choice(f'layer_{0}_units', values = [256, 64, 128]), kernel_initializer = 'he_uniform'))
		model.add(layers.ELU())
	hp_num_hidden = hp.Int('units', min_value = 0, max_value = 4)
	for i in range(hp_num_hidden):
		j = i + 1
		if hp_act_fn == 'softplus':
			model.add(Dense(hp.Choice(f'layer_{j}_units', values = [16,32, 64, 128])))
			model.add(layers.Activation('softplus'))
		elif hp_act_fn == 'ssp':
			model.add(Dense(hp.Choice(f'layer_{j}_units', values = [16,32, 64, 128]), activation = shifted_softplus))
		else:
			model.add(Dense(hp.Choice(f'layer_{j}_units', values = [16,32, 64, 128]), kernel_initializer = 'he_uniform'))
			model.add(layers.ELU())

	model.add(Dense(1))
	opt = optimizers.Adam(learning_rate = hp.Choice('learning_rate', [.01, .001, .0001]))
	model.compile(optimizer = opt, loss = 'mse')
	return model 

tuner = kt.RandomSearch(model_builder, 
			objective = 'val_loss',
			executions_per_trial = 1,
			max_trials = 50,
			directory = 'kt_results',
			project_name = "Random_Search_mbtr",
			overwrite = True,
			)

tuner_hb = kt.Hyperband(model_builder,
			objective = 'val_loss',
			directory = 'kt_results',
			project_name = 'hb_tune',
			factor = 3, 
			max_epochs = 80,
			)
					

tuner.search(D_train_stck, E_train_stck, callbacks = callbacks, validation_data = (D_val, E_val), epochs = 100, verbose = 1)

best_hps=tuner.get_best_hyperparameters()[0].values
print(best_hps)
print(tuner.results_summary())
print(tuner.get_best_models()[0].summary())	
	
	
