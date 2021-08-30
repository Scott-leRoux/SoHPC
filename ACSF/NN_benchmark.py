import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gc
import os
from tensorflow.keras import initializers 
from tensorflow.keras import optimizers
import numpy as np
import time 
import sys
D_train = np.load("acsf_f_train.npy")
D_test = np.load("acsf_f_test.npy")
D_val = np.load("acsf_f_val.npy")

E_train = np.load("energy_f_train.npy")
E_test = np.load("energy_f_test.npy")
E_val = np.load("energy_f_val.npy")
gpu = True
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
gpu = False

N_CPUS = int(sys.argv[1])
#Z_train = np.load("charges_train.npy")
n_features = D_train.shape[1]
n_energy_output = 1 
#n_charges_output = Z_train.shape[1]
outputs = []
tf.config.threading.set_intra_op_parallelism_threads(N_CPUS)

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


def train_model(model,inputs, outputs, batch_size, n_epochs, val_in, val_out):
	history = model.fit(inputs, outputs, validation_data = (val_in, val_out), epochs = n_epochs,
			batch_size = batch_size, verbose = 1)
	return model, history

top = []
top.append(128)
top.append(64)
top.append(32)

t0 = time.time()
model = make_model(n_features, 1, top)
model, history = train_model(model, D_train, E_train, 32, 300, D_val, E_val)
t1 = time.time()
with open('nn_bench.txt','a') as f:
	f.write("\n")
	if gpu:
		f.write(f"GPU time: {t1-t0}")
	else:
		f.write(f"CPU: {N_CPUS}, time: {t1-t0}")
		print(f"CPU time: {t1-t0}")
'''
model.save("model")
np.save("history", history.history)

test_loss = model.evaluate(D_test, E_test)
print("test_loss", test_loss) 
'''
