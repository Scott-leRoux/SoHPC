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
N_JOBS = int(sys.argv[2])

path = '/lustre/home/sleroux/SoHPC/data/'
file_path = path + str(sys.argv[1])

structures = read(file_path, index=":")


structures.sort(key = lambda x: x.info["U0"])
energies = []
charges = []
num_atoms = []

species = set() 
max_n_atoms = 0
for system in structures:
	energies.append(float(system.info["U0"]))
	atoms = system.get_chemical_symbols()
	species.update(atoms)
	if max_n_atoms < len(atoms):
		max_n_atoms = len(atoms)
	num_atoms.append(len(atoms))
print(max_n_atoms)
for system in structures:
	z = system.get_initial_charges()
	z = z.tolist()
	diff = max_n_atoms - len(z)
	padding = [0] * diff
	z.extend(padding)
	charges.append(z)
	


print("Number of systems: {}".format(len(structures)))


# Let's create a list of structures and gather the chemical elements that are
# in all the structures.

print("Creating descriptors...")

#print(species)


# Let's configure the SOAP descriptor.
from dscribe.descriptors import SOAP
import numpy as np 

soap = SOAP(
    species=species,
    periodic=False,
    rcut=5,
    nmax=8,
    lmax=8,
    average='outer',
    sparse=False
)

# Let's create SOAP feature vectors for each structure
print("creating features .....")
import time

#split data
t0 = time.time()
soap_qm9 = soap.create(structures, n_jobs = N_JOBS)
t1 = time.time()
t = t1 - t0
with open('test.txt', 'a') as f:
	print(f"no. of cores: {N_JOBS}, time: {t}")
	f.write(f"no. of cores: {N_JOBS}, time: {t}")
energies = np.array(energies)
print(charges[0])
charges = np.array(charges)
print(soap_qm9.shape)
print(energies.shape)
print(charges.shape)
kf = KFold(n_splits = 5, shuffle = True, random_state = 9)

D_test = soap_qm9[::5]
E_test = energies[::5]
Z_test = charges[::5]

soap_qm9_train = np.delete(soap_qm9, slice(None,None, 5), axis = 0)
energies = np.delete(energies, slice(None,None, 5))
charges = np.delete(charges, slice(None,None, 5), axis = 0)
print(soap_qm9_train.shape)
print(energies.shape)
print(charges.shape)
Z_train = []
E_train = []
D_train = [] 
for train, test in kf.split(soap_qm9_train):
	Z_train.append(charges[test])
	E_train.append(energies[test])
	D_train.append(soap_qm9_train[test])


for i in range(kf.get_n_splits()):
	np.save(f"soap_qm9_train{i}", D_train[i])
	np.save(f"charges_qm9_train{i}", Z_train[i])
	np.save(f"energies_qm9_train{i}", E_train[i])

np.save("soap_qm9_test", D_test)

np.save("energies_qm9_test", E_test)

np.save("charges_qm9_test", Z_test)

