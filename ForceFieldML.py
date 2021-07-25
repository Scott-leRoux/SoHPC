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

from sklearn.metrics import mean_absolute_error


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
D_train, D_val, E_train, E_val, F_train, F_val, dD_dr_train, dD_dr_val = train_test_split(
    D_train,
    E_train,
    F_train,
    dD_dr_train,
    test_size=.2,
    random_state = 7)


def make_model(n_features, n_hidden, n_out):
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev = 1.0)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_hidden, input_dim = n_features, activation = 'sigmoid', 
                                    kernel_initializer = initializer))
    model.add(tf.keras.layers.Dense(n_out, activation = 'linear', kernel_initializer = initializer))
    
    return model

@tf.function
def force_energy_loss(E_hat, F_hat, E, F):
    E_loss = tf.math.reduce_mean((E_hat - E)**2) / Evar_train
    F_loss = tf.math.reduce_mean((F_hat - F)**2) / Fvar_train
    return E_loss + F_loss




class FFnet(tf.keras.Model):
    
    def __init__(self, n_features, n_hidden, n_out):
        super(FFnet, self).__init__()
        ker_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)
        self.dense1 = tf.keras.layers.Dense(n_hidden, input_dim = n_features, 
                                            kernel_initializer=ker_init, 
                                            activation = 'sigmoid')
        self.dense2 = tf.keras.layers.Dense(n_out, activation = 'linear')
        
        
    def call(self, inputs):
        
        x = self.dense1(inputs)
        return self.dense2(x)



model = make_model(n_features, 5, 1)
ffmodel = FFnet(n_features, 5, 1)


opt = tf.keras.optimizers.Adam(lr = .001)

   

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

E_train_tf = tf.constant(E_train)
E_val_tf = tf.constant(E_val)

F_train_tf = tf.constant(E_train)
F_val_tf = tf.constant(F_val)

dD_dr_train_tf, dD_dr_val_tf = tf.constant(dD_dr_train), tf.constant(dD_dr_val)

perm = [i for i in range(len(D_train_tf))]
perm = tf.random.shuffle(perm)


@tf.function
def calc_gradients(D_train_batch, E_train_batch, F_train_batch, opt):
     with tf.GradientTape() as tape1:
          with tf.GradientTape() as tape2:
              tape2.watch(D_train_batch)
              E_pred = model(D_train_batch, training=True)
                
            
          df_dD_train_batch = tape2.gradient(E_pred, D_train_batch)
          df_dD_train_batch = tf.cast(df_dD_train_batch, dtype=tf.float64)
            
          #matrix mult of -Grad_D(f) x Grad_r(D)
          F_pred = -tf.einsum('ijkl,il->ijk', dD_dr_train_batch, df_dD_train_batch)
            
          F_pred = tf.cast(F_pred, dtype=tf.float32)
          E_pred = tf.cast(E_pred, dtype=tf.float32)
          E_train_batch = tf.cast(E_train_batch, dtype=tf.float32)
          F_train_batch = tf.cast(F_train_batch, dtype=tf.float32)
            
          #calculate loss value
          loss = force_energy_loss(E_pred, F_pred, E_train_batch, F_train_batch)
      
     grads = tape1.gradient(loss, model.trainable_weights)
     opt.apply_gradients(zip(grads,model.trainable_weights))
     
    
@tf.function
def calc_val_loss(E_val_tf, F_val_tf):
    with tf.GradientTape() as val_tape:
        val_tape.watch(D_val_tf)
        #set training = False, to ensure the weights aren't updated during validation stage
        E_pred_val = model(D_val_tf, training=False)
            
            
    #get gradients of output w.r.t input and cast       
    df_dD_val = val_tape.gradient(E_pred_val, D_val_tf)       
    df_dD_val = tf.cast(df_dD_val, dtype=tf.float64)
            
             
    F_pred_val = -tf.einsum('ijkl,il->ijk', dD_dr_val_tf, df_dD_val)
            
    F_pred_val = tf.cast(F_pred_val, dtype=tf.float32)
    E_pred_val = tf.cast(E_pred_val, dtype=tf.float32)
            
    E_val_tf = tf.cast(E_val_tf, dtype=tf.float32)
    F_val_tf = tf.cast(F_val_tf, dtype=tf.float32)
            
    return force_energy_loss(E_pred_val, F_pred_val, E_val_tf, F_val_tf)      
            
        
        
        
           
        
    

for i in range(n_max_epochs):
    
    #batches 
    perm = tf.random.shuffle(perm)
    for j in range(0,len(D_train_tf), batch_size):
        indices = perm.numpy()[j:j+batch_size]
        
        D_train_batch, E_train_batch = tf.gather(D_train_tf,indices), tf.gather(E_train_tf,indices)
        F_train_batch, dD_dr_train_batch = tf.gather(F_train_tf, indices), tf.gather(dD_dr_train, indices)
        """ 
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
            df_dD_train_batch = tf.cast(df_dD_train_batch, dtype=tf.float64)
            
            #matrix mult of -Grad_D(f) x Grad_r(D)
            F_pred = -tf.einsum('ijkl,il->ijk', dD_dr_train_batch, df_dD_train_batch)
            
            
            #casting tensors to floats 
            F_pred = tf.cast(F_pred, dtype=tf.float32)
            E_pred = tf.cast(E_pred, dtype=tf.float32)
            E_train_batch = tf.cast(E_train_batch, dtype=tf.float32)
            F_train_batch = tf.cast(F_train_batch, dtype=tf.float32)
            
            #calculate loss value
            loss = force_energy_loss(E_pred, F_pred, E_train_batch, F_train_batch)
        
        #get gradient of loss w.r.t to trainable weights for back propogation
        grads = outer.gradient(loss, model.trainable_weights)
        
        #updates weights using the optimizer and the gradients (grads)
        opt.apply_gradients(zip(grads, model.trainable_weights))
        """
        calc_gradients(D_train_batch, E_train_batch, F_train_batch, opt)
        
        """
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                tape2.watch(D_train_batch)
                E_pred = model(D_train_batch, training=True)
            
            df_dD_train_batch = tape2.gradient(E_pred, D_train_batch)
            df_dD_train_batch = tf.cast(df_dD_train_batch, dtype=tf.float64)
            
            #matrix mult of -Grad_D(f) x Grad_r(D)
            F_pred = -tf.einsum('ijkl,il->ijk', dD_dr_train_batch, df_dD_train_batch)
            
            F_pred = tf.cast(F_pred, dtype=tf.float32)
            E_pred = tf.cast(E_pred, dtype=tf.float32)
            E_train_batch = tf.cast(E_train_batch, dtype=tf.float32)
            F_train_batch = tf.cast(F_train_batch, dtype=tf.float32)
            
            #calculate loss value
            loss = force_energy_loss(E_pred, F_pred, E_train_batch, F_train_batch)
        
        #get gradient of loss w.r.t to trainable weights for back propogation
        grads = tape1.gradient(loss, model.trainable_weights)
        
        
        #updates weights using the optimizer and the gradients (grads)
        opt.apply_gradients(zip(grads, tw))
        """
        
        
        """
        if i % 4 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (i, float(loss))
            )
            print("Seen so far: %s samples" % ((i + 1) * batch_size))
         """

    print('------------Epoch %d -------------'%(i))
    #validation stage
    """
    with tf.GradientTape() as val_tape:
        val_tape.watch(D_val_tf)
        #set training = False, to ensure the weights aren't updated during validation stage
        E_pred_val = model(D_val_tf, training=False)
            
            
    #get gradients of output w.r.t input and cast       
    df_dD_val = val_tape.gradient(E_pred_val, D_val_tf)       
    df_dD_val = tf.cast(df_dD_val, dtype=tf.float64)
            
             
    F_pred_val = -tf.einsum('ijkl,il->ijk', dD_dr_val_tf, df_dD_val)
            
    F_pred_val = tf.cast(F_pred_val, dtype=tf.float32)
    E_pred_val = tf.cast(E_pred_val, dtype=tf.float32)
            
    E_val_tf = tf.cast(E_val_tf, dtype=tf.float32)
    F_val_tf = tf.cast(F_val_tf, dtype=tf.float32)
            
    val_loss = force_energy_loss(E_pred_val, F_pred_val, E_val_tf, F_val_tf)
    """
    val_loss = calc_val_loss(E_val_tf, F_val_tf)
    if val_loss < best_valid_loss:
        
        print("saving model at Epoch {}".format(i))
        model.save('model')
        best_valid_loss = val_loss
    if val_loss >= old_valid_loss:
        i_worse += 1
    else:
        i_worse = 0
    
    if i_worse > patience:
        print("Early stopping at Epoch {}".format(i))
        break
    old_valid_loss = val_loss
    

#enter evaluation stage

#load best model 
final_model = load_model('model')

#create tf.tensors for the whole spaces
D_tf = tf.constant(D_np)
F_tf = tf.constant(F_np)
E_tf = tf.constant(E_np)
r_tf = tf.constant(r_np)
dD_dr_tf = tf.constant(dD_dr)

with tf.GradientTape() as test_tape:
    test_tape.watch(D_tf)
    E_pred_test = final_model(D_tf, training=False)

df_dD_test = test_tape.gradient(E_pred_test, D_tf)
df_dD_test = tf.cast(df_dD_test, dtype = tf.float64)

F_pred_test = -tf.einsum('ijkl,il->ijk', dD_dr_tf, df_dD_test)

F_tf = tf.cast(F_tf, dtype = tf.float32)
E_tf = tf.cast(E_tf, dtype = tf.float32)

F_pred_test = tf.cast(F_pred_test, dtype = tf.float32)
E_pred_test = tf.cast(E_pred_test, dtype = tf.float32)

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

ax2.text(0.95, .5, "MAE: {:.2} eV".format(mae_F),  
         horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes)

ax1.legend(fontsize=12)
plt.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.08, hspace=0)
plt.show()







    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        