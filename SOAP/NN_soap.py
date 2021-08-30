import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras import initializers 
from tensorflow.keras import optimizers
import numpy as np
D_train = np.load("soap_train.npy")
D_test = np.load("soap_test.npy")
D_val = np.load("soap_valid.npy")

E_train = np.load("energies_train.npy")
E_test = np.load("energies_test.npy")
E_val = np.load("energies_valid.npy")

Z_train = np.load("charges_train.npy")
n_features = D_train.shape[1]
n_energy_output = 1 
n_charges_output = Z_train.shape[1]
outputs = []
print(D_train.shape)
print(E_train.shape)
def make_model(n_features, n_output, net_topology, lr = .001, batch_size=32, n_epochs= 100):
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

def train_model(model, batch_size, n_epochs, features, val_in, val_out):
	history = model.fit(features,E_train, validation_data = (val_in, val_out), epochs = n_epochs,
			batch_size = batch_size, verbose = 0)
	return model, history

top = []
top.append(100)
top.append(30)
top.append(50)

model = make_model(n_features, n_energy_output, top)
model, history = train_model(model, 32, 100, D_train, D_val, E_val)



model.save("model")
np.save("history", history.history)

test_loss = model.evaluate(D_test, E_test)
print("test_loss", test_loss) 
