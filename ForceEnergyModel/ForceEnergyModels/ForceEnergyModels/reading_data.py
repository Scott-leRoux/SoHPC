"""
Created on Tue Aug 10 14:12:52 2021

@author: scottleroux
"""
"""
Created on Wed Jul 14 18:37:36 2021

@author: scottleroux
"""

from ase.db import connect
from dscribe.descriptors import SOAP 
import numpy as np

forces = []
energy = []

forces_t1, forces_t2 = [], []
energy_t1, energy_t2 = [], []

structures = []
structures_t1 = []
structures_t2 = []


species = set()
db = connect('reference.db')
for row in db.select():
       structures.append(row.toatoms())
       species.update(row.toatoms().get_chemical_symbols())
       energy.append(row['total_energy'])
       forces.append(row.data['atomic_forces'])
            
db_test1 = connect('test_within.db')
db_test2 = connect('test_other.db')

    
for row in db_test1.select():
       structures_t1.append(row.toatoms())
       species.update(row.toatoms().get_chemical_symbols())
       energy_t1.append(row['total_energy'])
       forces_t1.append(row.data['atomic_forces'])      
       
for row in db_test2.select():
       structures_t2.append(row.toatoms())
       species.update(row.toatoms().get_chemical_symbols())
       energy_t2.append(row['total_energy'])
       forces_t2.append(row.data['atomic_forces'])      
             
   
print('done')
with open('train_ids.txt') as f:
    lines = f.readlines()
    
train_ind = []

for i in lines:
    train_ind.append(int(i)-1)
    
with open('validation_ids.txt') as f:
    lines = f.readlines()
    
val_ind = []

for i in lines:
    val_ind.append(int(i)-1)
     

soap = SOAP(species = species,
            periodic = False,
            rcut = 5.0,
            sigma = .5,
            nmax = 3,
            lmax = 0)

derivatives, descriptor = soap.derivatives(structures, n_jobs = 1, method = 'analytical')
dr_t1, desc_t1 = soap.derivatives(structures_t1, n_jobs = 1, method = 'analytical')
dr_t2, desc_t2 = soap.derivatives(structures_t2, n_jobs = 1, method = 'analytical')

n = descriptor.shape[1]
m = descriptor.shape[2]
d = descriptor.reshape((-1,n*m))
dt1 = desc_t1.reshape((-1,n*m))
dt2 = desc_t2.reshape((-1,n*m))

n1 = derivatives.shape[1]
m1 = derivatives.shape[4]
dD_dr = derivatives.transpose(0,2,3,1,4)
dD_dr = dD_dr.reshape(dD_dr.shape[0], dD_dr.shape[1], dD_dr.shape[2], n1*m1)

dD_dr_t1 = dr_t1.transpose(0,2,3,1,4)
dD_dr_t1 = dD_dr_t1.reshape(dD_dr_t1.shape[0], dD_dr_t1.shape[1], dD_dr_t1.shape[2], n1*m1)

dD_dr_t2 = dr_t2.transpose(0,2,3,1,4)
dD_dr_t2 = dD_dr_t2.reshape(dD_dr_t2.shape[0], dD_dr_t2.shape[1], dD_dr_t2.shape[2], n1*m1)



forces = np.array(forces)
forces_val = forces[val_ind]
forces_train = forces[train_ind]

forces_t1 = np.array(forces_t1)
forces_t2 = np.array(forces_t2)

energy = np.array(energy)
energy_val = energy[val_ind]
energy_train = energy[train_ind]

energy_t1 = np.array(energy_t1)
energy_t2 = np.array(energy_t2)

D_train = d[val_ind]
D_val = d[train_ind]

dD_dr_train = dD_dr[train_ind]
dD_dr_val = dD_dr[val_ind]


np.save('F_train', forces_train)
np.save('F_val', forces_val)
np.save('F_test_in', forces_t1)
np.save('F_test_out', forces_t2)

np.save('E_train', energy_train)
np.save('E_val', energy_val)
np.save('E_test_in', energy_t1)
np.save('E_test_out', energy_t2)

np.save('D_train', D_val)
np.save('D_val', D_val)
np.save('D_test_in', dt1)
np.save('D_test_out', dt2)

np.save('dD_dr_train', dD_dr_val)
np.save('dD_dr_val', dD_dr_val)
np.save('dD_dr_test_in', dD_dr_t1)
np.save('dD_dr_test_out', dD_dr_t2)



