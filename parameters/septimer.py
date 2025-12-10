# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:17:32 2020
this file creates arrays for the dipole orientations and interaction hamiltonian 
for the 7-mer 'septimer' used in Herman's paper.
note that the basis order is different than the papers.

the dipoles are in units of debye.  The Hamiltonian has units of cm^-1. 

"Single-photon absorption by single photosynthetic light-harvesting complexes"
H. C. H. Chan, O. E. Gamel, G. R. Fleming, and K. B. Whaley, J. Phys. B: At. Mol. Opt. Phys. 51, 054002 (2018)

@author: rlcook
"""

from numpy import array

names = array(['CLA C 602', 'CLA C 603', 'CHL C 608', 'CHL C 609', 'CLA C 610',
       'CLA C 611', 'CLA C 612'])


lambdas = array([37., 37., 37., 37., 37., 37., 37.])

gammas = array([30., 30., 48., 48., 30., 30., 30.])

dipoles = array([[ 3.70655145,  0.49853347,  1.41878142],
       [-3.54931094,  0.93774289,  1.5884049 ],
       [ 2.41712577, -2.04852184,  1.23331315],
       [ 3.34545892, -0.47923412,  0.37180543],
       [-3.72776659, -0.10277908,  1.44678703],
       [-3.92795873,  0.71899245,  0.23278755],
       [ 3.19528041, -1.00319385,  2.18718659]])


H_sys = array([[ 1.5157e+04,  3.8110e+01, -5.8400e+00, -1.9250e+01, -1.1390e+01, 9.6900e+00,  1.5830e+01],
       [ 3.8110e+01,  1.5287e+04,  6.7200e+00,  9.6660e+01,  1.2970e+01, -2.7000e+00, -7.6000e-01],
       [-5.8400e+00,  6.7200e+00,  1.5761e+04,  3.6070e+01,  6.1970e+01, 4.3500e+00, -1.0800e+00],
       [-1.9250e+01,  9.6660e+01,  3.6070e+01,  1.5721e+04,  3.8600e+00, 4.3000e+00, -2.5700e+00],
       [-1.1390e+01,  1.2970e+01,  6.1970e+01,  3.8600e+00,  1.5073e+04, -2.4960e+01,  2.3100e+01],
       [ 9.6900e+00, -2.7000e+00,  4.3500e+00,  4.3000e+00, -2.4960e+01, 1.5112e+04,  1.2692e+02],
       [ 1.5830e+01, -7.6000e-01, -1.0800e+00, -2.5700e+00,  2.3100e+01, 1.2692e+02,  1.5094e+04]])

# singular vals = sqrt [85.63333589,  4.28800945, 13.19865466]
# SVD polarizations
# [ 0.98250733, -0.17345777,  0.06776239]
# [-0.15558759, -0.96455265, -0.21314473]
# [-0.10233201, -0.19887327,  0.97466793]

import numpy as np
from numpy import matmul
polarization = np.array([ 0.98250733, -0.17345777,  0.06776239])
B_unnorm = matmul(dipoles, polarization)
B_norm = B_unnorm/np.linalg.norm(B_unnorm)
ctrfreq = np.average(np.diag(H_sys))
I = np.identity(7)
eigvals, eigvecs = np.linalg.eig(H_sys-ctrfreq*I)
overlap_sq = np.square(np.abs(matmul(B_norm, eigvecs)))
energy_sq = np.square(eigvals)
delta = np.sqrt(matmul(overlap_sq, energy_sq))
print('delta =', str(delta))
