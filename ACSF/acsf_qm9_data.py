#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:07:11 2021

@author: scottleroux
"""

################### dscribe part #################

from ase.io import read
import sys
import math
# Let's use ASE to create atomic structures as ase.Atoms objects.
print("Loading data...")


path = '/lustre/home/sleroux/SoHPC/data/'
file_path = path + str(sys.argv[1])
file_path2 = path + str(sys.argv[2])
file_path3 = path + str(sys.argv[3])

structures = read(file_path, index=":")
test_structures = read(file_path2, index=":")
val_structures = read(file_path3, index=":")


energies = []
charges = []
species = set() 
max_dist = 0
max_n_atoms = 0

for system in structures:
        energies.append(float(system.info["U0"]))
        atoms = system.get_chemical_symbols()
        species.update(atoms)
        if max_n_atoms < len(atoms):
                max_n_atoms = len(atoms)
        d = max(system.get_all_distances().flatten())
        if d > max_dist:
                max_dist = d

energies_v = []
for system in val_structures:
        energies_v.append(float(system.info["U0"]))
        atoms = system.get_chemical_symbols()
        species.update(atoms)
        if max_n_atoms < len(atoms):
                max_n_atoms = len(atoms)
        d = max(system.get_all_distances().flatten())
        if d > max_dist:
                max_dist = d


energies_t = []
for system in test_structures:
	energies_t.append(float(system.info["U0"]))
	atoms = system.get_chemical_symbols()
	species.update(atoms)
	if max_n_atoms < len(atoms):
		max_n_atoms = len(atoms)
	d = max(system.get_all_distances().flatten())
	if d > max_dist:
		max_dist = d
max_d = math.ceil(max_dist)

#flatten charges
'''
for system in structures:
	z = system.get_initial_charges()
	z = z.tolist()
	diff = max_n_atoms - len(z)
	padding = [0] * diff
	z.extend(padding)
	charges.append(z)
'''	


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
	rcut=max_d,
	g2_params = [[1,2], [1,3], [1,1]],
	g3_params = None,
	g4_params = [[1, 4, 1], [0.1, 4, 1], [0.01, 4, 1],
			[1, 4, -1], [0.1, 4, -1], [0.01, 4, -1]],
	g5_params = None	
)

num_ft = acsf.get_number_of_features()
# Let's create SOAP feature vectors for each structure
print("creating features .....")
print(num_ft)

#split data
acsf_qm9 = acsf.create(structures, n_jobs = 16)
acsf_test = acsf.create(test_structures, n_jobs = 16)
acsf_v = acsf.create(val_structures, n_jobs = 16)

max_fv = num_ft * max_n_atoms
print(max_fv)
for i in range(len(acsf_qm9)):
	acsf_qm9[i].flatten()
	j = np.resize(acsf_qm9[i], (1,max_fv))
	acsf_qm9[i] = j		

for i in range(len(acsf_test)):
	acsf_test[i].flatten()
	j = np.resize(acsf_test[i], (1,max_fv))
	acsf_test[i] = j

for i in range(len(acsf_v)):
	acsf_v[i].flatten()
	j = np.resize(acsf_v[i], (1,max_fv))
	acsf_v[i] = j

acsf_qm9 = np.vstack(acsf_qm9)
acsf_test = np.vstack(acsf_test)
acsf_val = np.vstack(acsf_v)
print(acsf_qm9.shape)
energies = np.array(energies)
energies_t = np.array(energies_t)
energies_v = np.array(energies_v)

np.save("acsf_f_train", acsf_qm9)
np.save("acsf_f_test", acsf_test)
np.save("acsf_f_val", acsf_val)

np.save("energy_f_train", energies)
np.save("energy_f_val", energies_v)
np.save("energy_f_test", energies_t)



