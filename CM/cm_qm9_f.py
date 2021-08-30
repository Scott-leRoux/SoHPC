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


print("Number of systems: {}".format(len(structures)))


# Let's create a list of structures and gather the chemical elements that are
# in all the structures.

print("Creating descriptors...")

#print(species)


# Let's configure the SOAP descriptor.
from dscribe.descriptors import CoulombMatrix
import numpy as np 

cm = CoulombMatrix(
	n_atoms_max = max_n_atoms,
	flatten = True,
	)

# Let's create SOAP feature vectors for each structure
print("creating features .....")

#split data
cm_qm9 = cm.create(structures, n_jobs = 1)
cm_test = cm.create(test_structures, n_jobs = 1)
cm_val = cm.create(val_structures, n_jobs = 1)

energies = np.array(energies)
energies_t = np.array(energies_t)
energies_v = np.array(energies_v)

np.save("cm_f_train", cm_qm9)
np.save("cm_f_test", cm_test)
np.save("cm_f_val", cm_val)

np.save("energy_f_train", energies)
np.save("energy_f_val", energies_v)
np.save("energy_f_test", energies_t)



