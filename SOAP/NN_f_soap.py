import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras import initializers 
from tensorflow.keras import optimizers
import numpy as np

import os
import sys
D_train = np.load("soap_f_train.npy")
D_test = np.load("soap_f_test.npy")
D_val = np.load("soap_f_val.npy")

E_train = np.load("energy_f_train.npy")
E_test = np.load("energy_f_test.npy")
E_val = np.load("energy_f_val.npy")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
# Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
# Invalid device or cannot modify virtual devices once initialized.
    pass

N_CPUS = int(sys.argv[1])
#Z_train = np.load("charges_train.npy")
tf.config.threading.set_intra_op_parallelism_threads(N_CPUS)
n_features = D_train.shape[1]
n_energy_output = 1 
#n_charges_output = Z_train.shape[1]
outputs = []
print(D_train.shape)
print(E_train.shape)
def make_model(n_features, n_output, net_topology, lr = .001):
	model = Sequential()
	elu = tf.keras.layers.ELU()
	model.add(Dense(net_topology[0], input_dim = n_features, 
			activation = elu, kernel_initializer = 'he_uniform'))
	
	for i in net_topology[1:]:
		model.add(Dense(i, activation = elu, kernel_initializer = 'he_uniform'))
	
	model.add(Dense(n_output, activation = 'linear'))
	opt = optimizers.Adam(learning_rate = lr)
	model.compile(loss = 'mse', optimizer = 'adam')
	return model



def train_model(model, inputs, outputs, batch_size, n_epochs, val_in, val_out, callbacks):
	history = model.fit(D_train,E_train, validation_data = (val_in, val_out), epochs = n_epochs,
			batch_size = batch_size, verbose = 1, callbacks = callbacks)
	return model, history

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience = 50)
mc = tf.keras.callbacks.ModelCheckpoint('model_soap', monitor='val_loss', mode='min', save_best_only=True)
callbacks = [es,mc]

top = []
top.append(256)
top.append(128)
top.append(128)

model = make_model(n_features, n_energy_output, top, .001)
model, history = train_model(model,D_train, E_train, 32, 300, D_val, E_val, callbacks = callbacks)

from matplotlib import pyplot
pyplot.title('SOAP training losses')
pyplot.plot(history.history['loss'],label='train')
pyplot.plot(history.history['val_loss'],label='test')
pyplot.legend()
pyplot.savefig('soap_loss_train.png')


test_loss = model.evaluate(D_test, E_test)
print("test_loss", test_loss) 
