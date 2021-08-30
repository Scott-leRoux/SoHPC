#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:07:11 2021

@author: scottleroux
"""

################### dscribe part #################

from ase.io import read
import sys
# Let's use ASE to create atomic structures as ase.Atoms objects.
print("Loading data...")

structures = []
N_JOBS = int(sys.argv[2])

path = '/lustre/home/sleroux/SoHPC/data/'
file_path = path + str(sys.argv[1])

structures = read(file_path, index=":")


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
	


print("Number of systems: {}".format(len(structures)))


# Let's create a list of structures and gather the chemical elements that are
# in all the structures.

print("Creating descriptors...")

#print(species)


# Let's configure the SOAP descriptor.
from dscribe.descriptors import ACSF
import numpy as np 

acsf = ACSF(
	species=species,
	periodic=False,
	rcut=5,
	g2_params = [[1,1],[1,2],[1,3]],
	g4_params = [[1, 4,1], [0.1, 4,1], [0.01, 4, 1],
			[1, 4, -1], [0.1, 4, -1], [0.01, 4, -1]]
	
)

# Let's create SOAP feature vectors for each structure
print("creating features .....")
import time

#split data
t0 = time.time()
acsf_qm9 = acsf.create(structures, n_jobs = N_JOBS)
t1 = time.time()
t = t1 - t0
with open('benchmark.txt', 'a') as f:
	print(f"n_jobs: {N_JOBS}, time: {t}")
	f.write(f"n_jobs: {N_JOBS}, time: {t}")

"""
max_fv = max([len(i) for i in acsf_qm9])
max_fv_d = max([i.shape[0]*i.shape[1] for i in acsf_qm9])

print(max_fv)
print(type(acsf_qm9[0]))
print(acsf_qm9[0].shape)
for i in acsf_qm9:
	i.flatten()
	i.resize(*max_fv, refcheck = False)		
print(acsf_qm9[0])
acsf_qm9 = np.vstack(acsf_qm9)
print(acsf_qm9.shape)

energies = np.array(energies)
print(charges[0])
charges = np.array(charges)
print(energies.shape)
print(charges.shape)
kf = KFold(n_splits = 5, shuffle = True, random_state = 9)

D_test = acsf_qm9[::5]
E_test = energies[::5]
Z_test = charges[::5]
print(type(acsf_qm9))
acsf_qm9_train = np.delete(acsf_qm9, slice(None,None, 5), axis = 0)
print(type(acsf_qm9_train))
energies = np.delete(energies, slice(None,None, 5))
charges = np.delete(charges, slice(None,None, 5), axis = 0)
print(acsf_qm9_train.shape)
print(energies.shape)
print(charges.shape)
Z_train = []
E_train = []
D_train = [] 
for train, test in kf.split(acsf_qm9_train):
	Z_train.append(charges[test])
	E_train.append(energies[test])
	D_train.append(acsf_qm9_train[test])


for i in range(kf.get_n_splits()):
	np.save(f"acsf_qm9_train{i}", D_train[i])
	np.save(f"charges_qm9_train{i}", Z_train[i])
	np.save(f"energies_qm9_train{i}", E_train[i])

np.save("acsf_qm9_test", D_test)

np.save("energies_qm9_test", E_test)

np.save("charges_qm9_test", Z_test)
"""
