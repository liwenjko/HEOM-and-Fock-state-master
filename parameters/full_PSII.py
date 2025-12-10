#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:51:47 2021

@author: LiwenKo
"""

#from PSII import PSII
import numpy as np

#psii = PSII()
#H_sys = psii.H
#dipoles = psii.dipoles
#chl_names = np.array(psii.chl_names)
#lambdas = np.zeros(np.size(chl_names))
#gammas = np.zeros(np.size(chl_names))
#for j in range(len(chl_names)):
#    if chl_names[j][:3] == 'CLA':
#        lambdas[j] = 37
#        gammas[j] = 30
#    elif chl_names[j][:3] == 'CHL':
#        lambdas[j] = 37
#        gammas[j] = 48
        
H_sys = np.load('PSII_H.npy')
dipoles = np.load('PSII_dipoles.npy')
lambdas = np.load('lambdas.npy')
gammas = np.load('gammas.npy')