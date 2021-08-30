import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras import initializers 
from tensorflow.keras import optimizers
import numpy as np
import math
import os
import sys
D_train = np.load("cm_f_train.npy")
D_test = np.load("cm_f_test.npy")
D_val = np.load("cm_f_val.npy")

E_train = np.load("energy_f_train.npy")
E_test = np.load("energy_f_test.npy")
E_val = np.load("energy_f_val.npy")
tf.compat.v1.disable_eager_execution()
'''
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
'''
N_CPUS = int(sys.argv[1])
#Z_train = np.load("charges_train.npy")
#tf.config.threading.set_intra_op_parallelism_threads(N_CPUS)
n_features = D_train.shape[1]
n_energy_output = 1 
#n_charges_output = Z_train.shape[1]
outputs = []
print(D_train.shape)
n_data_points = D_train.shape[0]
print(E_train.shape)
def make_model(n_features, n_output, net_topology, lr = .001):
	model = Sequential()
	model.add(Dense(net_topology[0], input_dim = n_features, 
			activation = "softplus", kernel_initializer = 'he_uniform'))
	
	for i in net_topology[1:]:
		model.add(Dense(i, activation = 'softplus', kernel_initializer = 'he_uniform'))
	
	model.add(Dense(n_output, activation = 'linear'))
	opt = optimizers.Adam(learning_rate = lr)
	model.compile(loss = 'mse', optimizer = 'adam')
	return model



def train_model(model, inputs, outputs, batch_size, n_epochs, val_in, val_out, callbacks):
	history = model.fit(D_train,E_train, validation_data = (val_in, val_out), epochs = n_epochs,
			batch_size = batch_size, verbose = 1)
	return model, history

def prep_data(D_train, D_test, D_val, E_train, E_test, E_val):
	#convert numpy arrays to Tensors
	D_train_tf, E_train_tf = tf.constant(D_train), tf.constant(E_train)
	D_test_tf, E_test_tf = tf.constant(D_test), tf.constant(E_test)
	D_val_tf, E_val_tf = tf.constant(D_val), tf.constant(E_val)
	
	train_ds = tf.data.Dataset.from_tensor_slices((D_train_tf,E_train_tf))
	test_ds = tf.data.Dataset.from_tensor_slices((D_test_tf,E_test_tf))
	val_ds = tf.data.Dataset.from_tensor_slices((D_val_tf,E_val_tf))
	
	train_ds = train_ds.cache()
	val_ds = val_ds.cache()
	
	train_ds = train_ds.shuffle(buffer_size = 200).repeat().batch(32)
	train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
	
	val_ds = val_ds.batch(D_val.shape[0]).prefetch(1)
	return train_ds, test_ds, val_ds


es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience = 50)
mc = tf.keras.callbacks.ModelCheckpoint('model_cm', monitor='val_loss', mode='min', save_best_only=True)
callbacks = [es,mc]

top = []
top.append(128)
top.append(32)
top.append(16)

model2 = make_model(n_features, n_energy_output, top, .0001)
model = make_model(n_features, n_energy_output, top, .0001)
#model, history = train_model(model,D_train, E_train, 32, 300, D_val, E_val, callbacks = callbacks)
import time 
train_ds, test_ds, val_ds = prep_data(D_train, D_test, D_val, E_train, E_test, E_val)

t1 =time.time()
history = model.fit(train_ds, steps_per_epoch= math.ceil((D_train.shape[0])/32), 
			validation_data = val_ds,
			validation_steps = 1, 
			epochs = 100, verbose = 1) 
t2 = time.time()

model2, history2 = train_model(model2,D_train, E_train, 32, 100, D_val, E_val, callbacks = callbacks)
t3 = time.time()

print(f"Time Input Pipeline: {t2-t1}, \n Time Normal: {t3-t2}")
from matplotlib import pyplot
pyplot.title('ACSF training losses')
pyplot.plot(history.history['loss'],label='train')
pyplot.plot(history.history['val_loss'],label='test')
pyplot.legend()
pyplot.savefig('Acsf_loss_train.png')


test_loss = model.evaluate(D_test, E_test)
print("test_loss", test_loss) 
