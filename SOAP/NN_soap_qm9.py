import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, activations
from tensorflow.keras import initializers 
from tensorflow.keras import optimizers
import numpy as np
import keras_tuner as kt

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


D_train, E_train, Z_train = [], [], []

for i in range(1,5):
	D_train.append(np.load(f"soap_qm9_train{i}.npy"))
	E_train.append(np.load(f"energies_qm9_train{i}.npy"))
	E_train[i-1] = np.reshape(E_train[i-1], (-1,1))
	Z_train.append(np.load(f"charges_qm9_train{i}.npy"))
D_train_stck = np.vstack(D_train)
E_train_stck = np.vstack(E_train)
Z_train_stck = np.vstack(Z_train)

D_test = np.load("soap_qm9_test.npy")
D_val = np.load("soap_qm9_train0.npy")

E_test = np.load("energies_qm9_test.npy")
E_val = np.load("energies_qm9_train0.npy")

Z_test = np.load("charges_qm9_test.npy")
Z_val = np.load("charges_qm9_train0.npy")


n_features = D_train_stck.shape[1]
n_energy_output = 1 
n_charges_output = Z_train_stck.shape[1]
outputs = []

print(D_test.shape)

def make_model(n_features, n_output, net_topology, lr = .001):
	model = Sequential()
	model.add(Dense(net_topology[0], input_dim = n_features, 
			activation = "softplus", kernel_initializer = initializers.RandomUniform()))
	
	for i in net_topology[1:]:
		model.add(Dense(i, activation = 'softplus'))
	
	model.add(Dense(n_output, activation = 'linear'))
	opt = optimizers.Adam(learning_rate = lr)
	model.compile(loss = 'mse', optimizer = 'adam')
	return model

def tune_hyper_params():
	pass

def train_model(model,inputs, outputs, batch_size, n_epochs, val_in, val_out, callbacks):
	history = model.fit(inputs,outputs, validation_data = (val_in, val_out), epochs = n_epochs,
			batch_size = batch_size, verbose = 1, callbacks = callbacks)
	return model, history

top = []
top.append(128)
top.append(32)

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1 , patience = 45)
mc = ModelCheckpoint("best_model_1", monitor = 'val_loss', mode = 'min', save_best_only = True, verbose = 1)

callbacks = [es, mc]


model = make_model(n_features, n_energy_output, top)
model, history = train_model(model,D_train_stck, E_train_stck, 32, 400, D_val, E_val, callbacks=callbacks)



model = tf.keras.models.load_model("best_model_1")
np.save("history-" + str(top), history.history)

test_loss = model.evaluate(D_test, E_test)
print("test_loss", test_loss) 

	
	
