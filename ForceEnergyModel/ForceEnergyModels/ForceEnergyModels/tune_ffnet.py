"""
Created on Wed Jul 14 13:58:39 2021

@author: scottleroux
"""

import numpy as np
import tensorflow as tf
import ase
from ase.calculators.lj import LennardJones
import matplotlib.pyplot as plt
from dscribe.descriptors import SOAP

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense 
from sklearn.metrics import mean_absolute_error

import time
import math


tf.random.set_seed(2)
# Select equally spaced points for training





#train
#we explicitly require that the gradients should be calculated for the input variables
F_train = np.load('F_train.npy')
F_val = np.load('F_val.npy')
F_test = np.load('F_test_in.npy')
F_test2 = np.load('F_test_out.npy')

E_train = np.load('E_train.npy')
E_val = np.load('E_val.npy')
E_test = np.load('E_test_in.npy')
E_test2 = np.load('E_test_out.npy')

D_train = np.load('D_train.npy')
D_val = np.load('D_val.npy')
D_test = np.load('D_test_in.npy')
D_test2 = np.load('D_test_out.npy')

dD_dr_train = np.load('dD_dr_train.npy')
dD_dr_val = np.load('dD_dr_val.npy')
dD_dr_test = np.load('dD_dr_test_in.npy')
dD_dr_test2 = np.load('dD_dr_test_out.npy')
#TensorFlow doesn't automatically do this as its usually not necessary 
#We use the TensorFlow API GradientTape to pull these gradients 

Fvar_train = F_train.var()
Evar_train = E_train.var()


def build_model(hp):
	model = tf.keras.models.Sequential()
	model.add(hp.Int('layer1_size',1,5), input_dim = n_features, activation = 'softplus')
	model.add(Dense(1))
	return model

	
class MyTuner(kt.tuners.RandomSearch):
	def run_trial(self, trial, D_train, E_train, F_train, dD_dr_train, D_val, E_val, F_val, dD_dr_val, **kwargs):
		kwargs['batch_size'] = trial.hyperparameters.Choice('bs', [2])
		
		#@tf.function
		def force_energy_loss(E_hat, F_hat, E, F):
			E_loss = tf.math.reduce_mean((E_hat - E)**2) / Evar_train
			F_loss = tf.math.reduce_mean((F_hat - F)**2) / Fvar_train
			return E_loss + F_loss

		model = self.hypermodel.build(trial.hyperparameters)
		lr = trial.hyperparameters.Choice('learn_rate', [.001])
		opt = tk.keras.optimizers.Adam(lr)
		batch_size = kwargs['batch_size']
		epoch_loss_metric = tf.keras.Mean()
		
		def calc_gradients(D_train_batch, E_train_batch, F_train_batch, dD_dr_train_batch, opt):
			#and d(output)/d(input)
			with tf.GradientTape() as tape1:
				with tf.GradientTape() as tape2:
					tape2.watch(D_train_batch)
					E_pred = model(D_train_batch, training=True)
				df_dD_train_batch = tape2.gradient(E_pred, D_train_batch)
      
				F_pred = -tf.einsum('ijkl,il->ijk', dD_dr_train_batch, df_dD_train_batch)
        
				loss = force_energy_loss(E_pred, F_pred, E_train_batch, F_train_batch)

			grads = tape1.gradient(loss, model.trainable_variables)
			opt.apply_gradients(zip(grads, model.trainable_variables))
			epoch_loss_metric.update_state(loss)
			return loss

		def one_step(indices,D_train, E_train, F_train, dD_dr_train):
			
			D_train_batch, E_train_batch = tf.gather(D_train,indices), tf.gather(E_train,indices)
			F_train_batch, dD_dr_train_batch = tf.gather(F_train, indices), tf.gather(dD_dr_train, indices)
			loss = calc_gradients(D_train_batch, E_train_batch, F_train_batch, dD_dr_train_batch, opt)
			return loss

		def calc_val_loss(D_val, E_val, F_val, dD_dr_val):
			with tf.GradientTape() as val_tape:
        			val_tape.watch(D_val)
        			E_pred_val = model(D_val, training=False)

    			df_dD_val = val_tape.gradient(E_pred_val, D_val)
    			F_pred_val = -tf.einsum('ijkl,il->ijk', dD_dr_val, df_dD_val)
    			return force_energy_loss(E_pred_val, F_pred_val, E_val, F_val)

		perm = [i for i in range(len(D_train))]
		perm = tf.random.shuffle(perm)

		for i in range(2):
			print(f"Epoch: {i}") 
			self.on_epoch_begin(trial, model, i, logs={})
        		perm = tf.random.shuffle(perm)
        		for j in range(0,len(D_train), batch_size):
				self.on_batch_begin(trial, model, j, logs={})
            			lst = [q for q in range(j,j+batch_size)]
            			indices = tf.gather(perm, lst).numpy()
            			batch_loss = one_step(indices, D_train, E_train, F_train, dD_dr_train)
				self.on_batch_end(trial, model, j, logs={"loss": batch_loss})
				
        		val_loss = calc_val_loss(E_val, F_val, D_val, dD_dr_val)
        		if val_loss < best_valid_loss:
            			best_valid_loss = val_loss
        		if val_loss >= old_valid_loss:
            			i_worse += 1
        		else:
            			i_worse = 0
        		if i_worse > patience:
            			tf.print("Early stopping at Epoch {}".format(i))
            			break
        		old_valid_loss = val_loss
			self.on_epoch_end(trial, model, i, logs={"loss": loss, "val_loss": val_loss}
			epoch_loss_metric.reset_states()
		
		




n_max_epochs = 5000
patience = 20
i_worse = 0

old_valid_loss = float("Inf")
best_valid_loss = float("Inf")

D_train_tf = tf.constant(D_train)
D_val_tf = tf.constant(D_val)

D_train_tf = tf.cast(D_train_tf, dtype = tf.float32)
D_val_tf = tf.cast(D_val_tf, dtype = tf.float32)

E_train_tf = tf.constant(E_train)
E_val_tf = tf.constant(E_val)

E_train_tf = tf.cast(E_train_tf, dtype = tf.float32)
E_val_tf = tf.cast(E_val_tf, dtype = tf.float32)

F_train_tf = tf.constant(F_train1)
F_val_tf = tf.constant(F_val)

F_train_tf = tf.cast(F_train_tf, dtype = tf.float32)
F_val_tf = tf.cast(F_val_tf, dtype = tf.float32)

dD_dr_train_tf, dD_dr_val_tf = tf.constant(dD_dr_train), tf.constant(dD_dr_val)

dD_dr_train_tf = tf.cast(dD_dr_train_tf, dtype = tf.float32)
dD_dr_val_tf = tf.cast(dD_dr_val_tf, dtype = tf.float32)


perm = [i for i in range(len(D_train_tf))]
perm = tf.random.shuffle(perm)


'''
@tf.function
def calc_gradients(D_train_batch, E_train_batch, F_train_batch, dD_dr_train_batch, opt):
    #set up gradient tape scope in order to track gradients of both d(Loss)/d(Weights)
    #and d(output)/d(input)
     with tf.GradientTape() as tape1:
          with tf.GradientTape() as tape2:
              #set gradient tape to watch Tensor
              tape2.watch(D_train_batch)
              #pass D thru model to get predicted energy vals
              E_pred = model(D_train_batch, training=True)        
          df_dD_train_batch = tape2.gradient(E_pred, D_train_batch) 
          #matrix mult of -Grad_D(f) x Grad_r(D)
          F_pred = -tf.einsum('ijkl,il->ijk',  dD_dr_train_batch, df_dD_train_batch)
          #calculate loss value
          loss = force_energy_loss(E_pred, F_pred, E_train_batch, F_train_batch)
     
     grads = tape1.gradient(loss, model.trainable_variables)
     opt.apply_gradients(zip(grads, model.trainable_variables))

@tf.function
def calc_val_loss(E_val_tf, F_val_tf, D_val_tf):
    with tf.GradientTape() as val_tape:
        val_tape.watch(D_val_tf)
        #set training = False, to ensure the weights aren't updated during validation stage
        E_pred_val = model(D_val_tf, training=False)
                 
    #get gradients of output w.r.t input and cast       
    df_dD_val = val_tape.gradient(E_pred_val, D_val_tf)       
                    
    F_pred_val = -tf.einsum('ijkl,il->ijk', dD_dr_val_tf, df_dD_val)
    return force_energy_loss(E_pred_val, F_pred_val, E_val_tf, F_val_tf) 
                   
@tf.function
def one_step(indices):
    #t1 = time.time()
    D_train_batch, E_train_batch = tf.gather(D_train_tf,indices), tf.gather(E_train_tf,indices)
    F_train_batch, dD_dr_train_batch = tf.gather(F_train_tf, indices), tf.gather(dD_dr_train_tf, indices)
    #before = time.time()
    #tf.print("time to parse = " + str(before - t1))
    calc_gradients(D_train_batch, E_train_batch, F_train_batch, dD_dr_train_batch, opt)
    #total = time.time() - before
    #tf.print("time on training step = " + str(total))


def train_model(perm, best_valid_loss, old_valid_loss, i_worse):
    for i in range(2):
        
        #batches 
        
        perm = tf.random.shuffle(perm)
        for j in range(0,len(D_train_tf), batch_size):
            #t1 = time.time()
            lst = [q for q in range(j,j+batch_size)]
            indices = tf.gather(perm, lst).numpy()
            one_step(indices)
            #t2 = time.time()
            #print("-------------" + str(t2 - t1))
        #validation stage
        val_loss = calc_val_loss(E_val_tf, F_val_tf, D_val_tf)
        val_losses.append(val_loss)
        if val_loss < best_valid_loss:
        
            tf.print("saving model at Epoch {}".format(i))
            before = time.time()
            model.save('model')
            total = time.time() - before
            tf.print("time on saving model = " + str(total))
            best_valid_loss = val_loss
        if val_loss >= old_valid_loss:
            i_worse += 1
        else:
            i_worse = 0
        if i_worse > patience:
            tf.print("Early stopping at Epoch {}".format(i))
            break
        old_valid_loss = val_loss
'''     
#enter evaluation stage
tuner = MyTuner(build_model, 
		objective = "val_loss",
		directory = 'tuner',
		project_name = 'test',
		max_trials = 2)

tuner.Search(D_train_tf, E_train_tf, F_train_tf, dD_dr_train_tf, D_val_tf, E_val_tf, F_val_tf, dD_dr_val_tf)        
