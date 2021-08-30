#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

#setting up the SOAP descriptor 
soap = SOAP(species = ["H"],
            periodic = False,
            rcut = 5.0,
            sigma = .5,
            nmax = 3,
            lmax = 0)


#Generate Dataset of Lennard-Jones energies and forces
n_samples = 200
traj = [] 
n_atoms = 2
energies = np.zeros(n_samples)
forces = np.zeros((n_samples,n_atoms, 3))

r = np.linspace(2.5, 5.0, n_samples)

for i,d in enumerate(r):
    a = ase.Atoms('HH', positions= [[-0.5 * d, 0, 0], [0.5 * d, 0, 0]])
    a.set_calculator(LennardJones(epsilon = 1.0, sigma = 2.9))
    traj.append(a)
    energies[i] = a.get_total_energy()
    forces[i,:,:] = a.get_forces() 
    
    
    

# Plot the energies to validate them
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
line, = ax.plot(r, energies)
plt.xlabel("Distance (Å)")
plt.ylabel("Energy (eV)")
plt.show()    


#create soap derivatives and descriptors for each sample
#one center is to be chosen to be directly between the atoms
derivatives, descriptors = soap.derivatives(
    traj,
    positions = [[[0,0,0]]] * len(r),
    method = "analytical")



D_np = descriptors[:,0,:]
n_samples, n_features = D_np.shape

E_np = np.array([energies]).T
F_np = forces

dD_dr = derivatives[:,0,:,:,:]
r_np = r

tf.random.set_seed(2)
# Select equally spaced points for training
n_train = 30
idx = np.linspace(0, len(r_np) - 1, n_train).astype(int)

D_train = D_np[idx]
E_train = E_np[idx]
F_train = F_np[idx]
dD_dr_train = dD_dr[idx]
r_train = r_np[idx]
scaler = StandardScaler().fit(D_train)
D_train = scaler.transform(D_train)
D_whole = scaler.transform(D_np)

dD_dr_train = dD_dr_train / scaler.scale_[None, None, None, :]
dD_dr = dD_dr / scaler.scale_[None,None, None, :]

#calculate variance of Energey and Forces in training set
Fvar_train = F_train.var()
Evar_train = E_train.var()


#split training into training and validation 
D_train, D_val, E_train, E_val, F_train1, F_val, dD_dr_train, dD_dr_val, r_train, r_val = train_test_split(
    D_train,
    E_train,
    F_train,
    dD_dr_train,
    r_train,
    test_size=.2,
    random_state = 7)

def make_model(n_features, n_hidden, n_out):
    ki1 = tf.keras.initializers.GlorotNormal()
    ki2 = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 1.0)
    kitest = tf.keras.initializers.Ones()
    lower1 = -1 / math.sqrt(1/n_features)
    lower2 = -1 / math.sqrt(1/n_hidden)
    bias_init = tf.keras.initializers.RandomUniform(lower1, -lower1)
    bias_init2 = tf.keras.initializers.RandomUniform(lower2, -lower2)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_hidden, input_dim = n_features, activation = 'sigmoid', 
                                    kernel_initializer = ki2,
                                    ))
    model.add(tf.keras.layers.Dense(n_out, activation = 'linear', 
                                    kernel_initializer = ki2,
                                                                        
                                    ))
    return model

@tf.function
def force_energy_loss(E_hat, F_hat, E, F):
    E_loss = tf.math.reduce_mean((E_hat - E)**2) / Evar_train
    F_loss = tf.math.reduce_mean((F_hat - F)**2) / Fvar_train
    return E_loss + F_loss


model = make_model(n_features, 5, 1)
opt = tf.keras.optimizers.Adam(learning_rate = 1e-2, epsilon = 1e-8)


#train
#we explicitly require that the gradients should be calculated for the input variables

#TensorFlow doesn't automatically do this as its usually not necessary 
#We use the TensorFlow API GradientTape to pull these gradients 

n_max_epochs = 5000
batch_size = 2
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

val_losses = []



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
def calc_gradients_persistent(D_train_batch, E_train_batch, F_train_batch, dD_dr_train_batch, opt):
    #set up gradient tape scope in order to track gradients of both d(Loss)/d(Weights)
        #and d(output)/d(input)
        with tf.GradientTape(persistent = True) as outer:
            
            #set gradient tape to watch Tensor
            outer.watch(D_train_batch)
            
            #output values from model, set trainable to be true to get 
            #model.trainable_weights out
            E_pred = model(D_train_batch, training=True)
            
            #set gradient tape to watch trainable weights
            outer.watch(model.trainable_weights)
            
            #get gradient of output (f/E_pred) w.r.t input (D/D_train_batch) and cast to double
            df_dD_train_batch = outer.gradient(E_pred, D_train_batch)
            
            #matrix mult of -Grad_D(f) x Grad_r(D)
            F_pred = -tf.einsum('ijkl,il->ijk', dD_dr_train_batch, df_dD_train_batch)

            #calculate loss value
            loss = force_energy_loss(E_pred, F_pred, E_train_batch, F_train_batch)
        
        #get gradient of loss w.r.t to trainable weights for back propogation
        grads = outer.gradient(loss, model.trainable_variables)
        #updates weights using the optimizer and the gradients (grads)
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
 
    for i in range(1):
    
        #batches 
        for j in range(0,len(D_train_tf), 24):
            indices = test_indices
            print(indices)
        
            D_train_batch, E_train_batch = tf.gather(D_train_tf,indices), tf.gather(E_train_tf,indices)
            F_train_batch, dD_dr_train_batch = tf.gather(F_train_tf, indices), tf.gather(dD_dr_train_tf, indices)
            calc_gradients_t(D_train_batch, E_train_batch, F_train_batch, dD_dr_train_batch, opt)

        

        tf.print('------------Epoch %d -------------'%(i))
        #validation stage
        val_loss = calc_val_loss(E_val_tf, F_val_tf, D_val_tf)
        print(val_loss)
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
        
t0 = time.time()
train_model(perm, best_valid_loss, old_valid_loss, i_worse)
#enter evaluation stage
t1 = time.time()
print("Training time = " + str(t1 - t0))

#load best model 
final_model = load_model('model')


#create tf.tensors for the whole spaces
D_tf = tf.constant(D_whole)
D_tf = tf.cast(D_tf, tf.float32)
F_tf = tf.constant(F_np)
F_tf = tf.cast(F_tf, tf.float32)
E_tf = tf.constant(E_np)
E_tf = tf.cast(E_tf, tf.float32)
r_tf = tf.constant(r_np)
r_tf = tf.cast(r_tf, tf.float32)
dD_dr_tf = tf.constant(dD_dr)
dD_dr_tf = tf.cast(dD_dr_tf, tf.float32)

with tf.GradientTape() as test_tape:
    test_tape.watch(D_tf)
    E_pred_test = final_model(D_tf, training=False)

df_dD_test = test_tape.gradient(E_pred_test, D_tf)


F_pred_test = -tf.einsum('ijkl,il->ijk', dD_dr_tf, df_dD_test)

F = F_tf.numpy()
E = E_tf.numpy()

E_h = E_pred_test.numpy()
F_h = F_pred_test.numpy()

order = np.argsort(r)

fig, (ax1,ax2) = plt.subplots(2, 1, sharex = True, figsize = (10,10))
ax1.plot(r[order], E[order], label = "True", linewidth = 3, linestyle = '-')
ax1.plot(r[order], E_h[order], label = "Predicted", linewidth=3, linestyle="-")
ax1.set_ylabel("Energy", size = 15)
mae_E = mean_absolute_error(E,E_h)
ax1.text(0.95, .5, "MAE: {:.2} eV".format(mae_E),  
         horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes)

F_h = F_h[order,0,0]
F = F[:,0,0][order]

ax2.plot(r[order], F[order], label = "True", linewidth = 3, linestyle = '-')
ax2.plot(r[order], F_h[order], label = "Predicted", linewidth = 3, linestyle = '-')
ax2.set_ylabel("Force", size = 15)

mae_F = mean_absolute_error(F,F_h)

ax2.text(0.95, .5, "MAE: {:.2} eV/Å".format(mae_F),  
         horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes)
F_train = F_train1[:, 0, 0]
ax1.scatter(r_train, E_train, marker="o", color="k", s=20, label="Training points", zorder=3)

ax2.scatter(r_train, F_train, marker="o", color="k", s=20, label="Training points", zorder=3)

ax1.legend(fontsize=12)
plt.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.08, hspace=0)
plt.show()



loss_v = []  

for i in range(len(val_losses)):
    loss_v.append(val_losses[i].numpy())


plt.plot(loss_v, label = "val loss")
plt.legend()
plt.show()






        
