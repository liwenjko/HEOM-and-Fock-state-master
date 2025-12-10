#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:04:09 2020

@author: LiwenKo
"""

from numpy import array

# in Debye
#dipoles = array([[4,0,0],
#                       [4, 0, 0]]) 
dipoles = array([[-1.42230404, -3.46070983,  1.41440402],
       [ 2.58738817,  2.60418809,  1.58859273]])
# in cm^-1
H_sys = array([[15287, 38.11],
                 [38.11, 15157]])
#H_sys = array([[15287, 0],
#                 [0, 15157]]) 

lambdas = array([37., 37.])

gammas = array([30., 30.])

from math import acos, sin, cos
import numpy as np
angle = acos(np.inner(dipoles[0], dipoles[1])/16)
dipoles = array([[0, 0, 1], [sin(angle), 0, cos(angle)]])*4
polarization1 = [sin(angle/2), 0, cos(angle/2)] # between dipoles
# [0.9090771423883736, 0, 0.4166278305478269]
polarization2 = [cos(angle/2), 0, -sin(angle/2)] # outside dipoles
# [0.4166278305478269, 0, -0.9090771423883736]

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#ax.quiver(0, 0, cos(angle/2), sin(angle/2), scale=5, color='b')
#ax.quiver(0, 0, cos(angle/2), -sin(angle/2), scale=5, color='b')
#ax.quiver(0, 0, 1, 0, scale=5, alpha=0.5, color='r')
#ax.quiver(0, 0, 0, 1, scale=5, alpha=0.5, color='r')

#import numpy as np
from numpy import matmul
polarization = np.array([0.4166278305478269, 0, -0.9090771423883736])
B_unnorm = matmul(dipoles, polarization)
B_norm = B_unnorm/np.linalg.norm(B_unnorm)
ctrfreq = np.average(np.diag(H_sys)) #+ 75.34833840238285
I = np.identity(2)
eigvals, eigvecs = np.linalg.eig(H_sys-ctrfreq*I)
overlap_sq = np.square(np.abs(matmul(B_norm, eigvecs)))
energy_sq = np.square(eigvals)
delta = np.sqrt(matmul(overlap_sq, energy_sq))
print('delta =', str(delta))

WWgamma_cm = 1.0719291822450668e-05
Gamma_inc = WWgamma_cm * matmul(B_unnorm, B_unnorm)