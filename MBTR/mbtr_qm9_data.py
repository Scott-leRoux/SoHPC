#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:07:11 2021

@author: scottleroux
"""

################### dscribe part #################

from ase.io import read
import sys
from tensorflow import keras
import keras_tuner as kt
import random 
from sklearn.model_selection import KFold, StratifiedKFold
# Let's use ASE to create atomic structures as ase.Atoms objects.
print("Loading data...")

structures = []


path = '/lustre/home/sleroux/SoHPC/data/'
file_path = path + str(sys.argv[1])

structures = read(file_path, index=":")


structures.sort(key = lambda x: x.info["U0"])
energies = []
charges = []
num_atoms = []

species = set() 
max_n_atoms = 0
max_atomic_val = 0
for system in structures:
	energies.append(float(system.info["U0"]))
	atoms = system.get_chemical_symbols()
	species.update(atoms)
	if max_n_atoms < len(atoms):
		max_n_atoms = len(atoms)
	num_atoms.append(len(atoms))
	atomic_v = max(system.get_atomic_numbers())
	if atomic_v > max_atomic_val:
		max_atomic_val = atomic_v

print(max_n_atoms)
for system in structures:
	z = system.get_initial_charges()
	z = z.tolist()
	diff = max_n_atoms - len(z)
	padding = [0] * diff
	z.extend(padding)
	print(z)
	charges.append(z)
	


print("Number of systems: {}".format(len(structures)))


# Let's create a list of structures and gather the chemical elements that are
# in all the structures.

print("Creating descriptors...")

#print(species)


# Let's configure the SOAP descriptor.
from dscribe.descriptors import MBTR
import numpy as np 

mbtr = MBTR(
	species = species,
	k1 = {
		"geometry": {"function": "atomic_number"},
		"grid": {"min": 0, "max": max_atomic_val,"n": 100, "sigma":.01},
	},
	k2 = {
		"geometry": {"function": "inverse_distance"},
		"grid": {"min": 0, "max": 2, "n": 100, "sigma": .01},
		"weighting": {"function": "exp", "scale": .3, "threshold":1e-3}
	},
	k3 = {
                "geometry": {"function": "cosine"},
                "grid": {"min": -1, "max": 1, "n": 100, "sigma": .01},
                "weighting": {"function": "exp", "scale": .3, "threshold":1e-3}
        },
	flatten = True,
	normalization = "l2_each"
)

# Let's create SOAP feature vectors for each structure
print("creating features .....")


#split data
mbtr_qm9 = mbtr.create(structures, n_jobs = 8)
energies = np.array(energies)
print(charges[0])
charges = np.array(charges)
print(mbtr_qm9.shape)
print(energies.shape)
print(charges.shape)
kf = KFold(n_splits = 5, shuffle = True, random_state = 9)

D_test = mbtr_qm9[::5]
E_test = energies[::5]
Z_test = charges[::5]

mbtr_qm9_train = np.delete(mbtr_qm9, slice(None,None, 5), axis = 0)
energies = np.delete(energies, slice(None,None, 5))
charges = np.delete(charges, slice(None,None, 5), axis = 0)
print(mbtr_qm9_train.shape)
print(energies.shape)
print(charges.shape)
Z_train = []
E_train = []
D_train = [] 
for train, test in kf.split(mbtr_qm9_train):
	Z_train.append(charges[test])
	E_train.append(energies[test])
	D_train.append(mbtr_qm9_train[test])


for i in range(kf.get_n_splits()):
	np.save(f"mbtr_qm9_train{i}", D_train[i])
	np.save(f"charges_qm9_train{i}", Z_train[i])
	np.save(f"energies_qm9_train{i}", E_train[i])

np.save("mbtr_qm9_test", D_test)

np.save("energies_qm9_test", E_test)

np.save("charges_qm9_test", Z_test)

