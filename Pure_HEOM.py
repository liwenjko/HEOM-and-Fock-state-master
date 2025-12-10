#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:38:12 2019

@author: LiwenKo
"""

from functools import lru_cache
from numpy import matmul
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from math import pi
from scipy.linalg import expm
from parameters.monomer import H_sys
import time

lvl = 15
Lambda = 37 # in invcm
gamma = 200 # in invcm
kT_K = 300.00 # in Kelvin
init_site = 1 # initial state's excited site
sections = 10 # number of piecewise integration sections

sites = np.shape(H_sys)[0] # Number of sites
dim = sites + 1
rhodim = dim * dim
cm_to_K = 1.4388
kT_cm = kT_K / cm_to_K 
cm_to_fs = 1e5/(6*pi) # the unit of time corresponding to 1 invcm, in fs
tf_fs = 4000 # in fs
timestep_fs = 1 # in fs 
tf_cm = tf_fs/cm_to_fs
timestep_cm = timestep_fs/cm_to_fs
ctrfreq = 15622 # in cm^-1. This parameter has no effect in pure HEOM calculation.
assert init_site <= sites
#################### Import Hamiltonian, set up initial state #################
print('Initializing...')
# ground state is indexed 0
H_sys_int = H_sys - np.identity(sites) * ctrfreq
H = np.zeros((dim, dim), dtype=complex)
H[1:, 1:] = H_sys_int
rho_init = np.zeros((dim, dim), dtype=complex)
#rho_init[init_site, init_site] = 1

rho_init[0,0] = 1
rho_init[1,0] = 0
rho_init[0,1] = 0
rho_init[1,1] = 0
H[1,0] = 22
H[0,1] = 22

assert np.shape(H) == (dim, dim)
########################### HEOM Indexing #####################################
@lru_cache(maxsize=9999)
def construct_HEOM_lvl_lst(fix_lvl, sites):
    if fix_lvl == 0 and sites == 0:
        return [[]]
    elif fix_lvl > 0 and sites == 0:
        return []
    else:
        lst = []
        for m in range(fix_lvl+1):
            temp = construct_HEOM_lvl_lst(fix_lvl-m, sites-1)
            sublst = []
            for t in temp:
                sublst.append([m]+t)
            lst.extend(sublst)
    return lst

HEOM_lvl_vecs = []
for v in range(lvl+1):
    HEOM_lvl_vecs.extend(construct_HEOM_lvl_lst(v, sites))        
HEOM_lvl_ind = {}
for i in range(len(HEOM_lvl_vecs)):
    temp = tuple(HEOM_lvl_vecs[i])
    HEOM_lvl_vecs[i] = temp
    HEOM_lvl_ind[temp] = i
totlvls = len(HEOM_lvl_vecs)
zerovec = tuple(np.zeros(sites, dtype=int))
HEOM_vec_len = totlvls * rhodim
HEOM_phys_ind = HEOM_lvl_ind[zerovec]

def getlevelvector(levelindex):
    return HEOM_lvl_vecs[levelindex]
def getlevelindex(levelvector):
    return HEOM_lvl_ind[levelvector]
def gethighervec(vec, site):
    ''' Returns the vector with site having one higher level. Returns None if
    the higher vector is beyond the HEOM level. '''
    temp = np.array(vec)
    temp[site-1] += 1
    temp2 = tuple(temp)
    if sum(temp2) <= lvl:
        return temp2
    else:
        return None
def getlowervec(vec, site):
    ''' Returns the vector with site having one lower level. Returns None if
    the lower vector has negative components. '''
    temp = np.array(vec)
    if temp[site-1] == 0:
        return None
    else:
        temp[site-1] -= 1
        return tuple(temp)
def HEOMsitelvl(vec, site):
    return vec[site-1]
def sumHEOMsitelvl(vec):
    return sum(vec[0:sites])

############## Methods for accessing rhos in vectorized HEOM state ############
def getrho(state, lvlvec):
    ''' Returns the auxiliary rho with index lvlvec 
    from vectorized HEOM state. '''
    temp = HEOM_lvl_ind[tuple(lvlvec)]
    return np.reshape(state[temp*rhodim: (temp+1)*rhodim], (dim,dim))
    
def addrho(state, lvlvec, rho):
    ''' Add rho to the auxiliary rho indexed lvlvec.
    rho is in matrix form with dimension (dim * dim)'''
    temp = np.reshape(rho, -1)
    temp2 = HEOM_lvl_ind[tuple(lvlvec)]
    state[temp2*rhodim: (temp2+1)*rhodim] += temp
    return None

def initHEOMvec(rhoinit):
    ''' Returns the vectorized HEOM state (with all the auxiliary rhos) '''
    rtn = np.zeros(HEOM_vec_len, dtype=complex)
    addrho(rtn, zerovec, rho_init)
    return rtn
################################ Useful functions #############################
def projection(site):
    '''returns the projection operator to single exciton state at site Site'''
    temp = np.zeros([dim, dim], dtype=complex)
    temp[site, site] = 1
    return temp

projs = [None]
for i in range(sites):
    projs.append(projection(i+1))

def commutator(A, B):
    return matmul(A, B) - matmul(B, A)
def anticommutator(A, B):
    return matmul(A, B) + matmul(B, A)
   
######################### Time derivative function ############################
def timederiv(t, state):
    rtnstate = np.zeros(HEOM_vec_len, dtype=complex)
    for v in HEOM_lvl_vecs:
        temp = -1j * commutator(H, getrho(state, v))
        temp -= sumHEOMsitelvl(v) * gamma * getrho(state, v)
        for s in range(1, sites+1):
            lowervec = getlowervec(v,s)
            highervec = gethighervec(v,s)
            if lowervec != None:
                temp2 = commutator(projs[s], getrho(state, lowervec))
                temp -= HEOMsitelvl(v,s) * 2 * Lambda * kT_cm * temp2
                temp2 = anticommutator(projs[s], getrho(state, lowervec))
                temp += HEOMsitelvl(v,s) * 1j * Lambda * gamma * temp2
            if highervec != None:
                temp += commutator(projs[s], getrho(state, highervec))
        addrho(rtnstate, v, temp)
    return rtnstate

########################## solve_dynamics Methods #############################
def solve_dynamics():
    starttime = time.time()
    print('solving ODE...')
    initstate = initHEOMvec(rho_init)
    tpoints = np.arange(0, tf_cm, timestep_cm)
    solution = solve_ivp(timederiv, (0, tf_cm), initstate, t_eval=tpoints)
    rhos = solution.y[HEOM_phys_ind*rhodim:(HEOM_phys_ind+1)*rhodim, :]
    tpoints = solution.t * cm_to_fs
    print('solve ODE time:', time.time()-starttime, 's')
    return tpoints, rhos
def solve_dynamics_pw():
    ''' Solve dynamics piecewise to reduce memory usage. '''
    starttime = time.time()
    print('solving ODE piecewise...')
    interval = tf_cm/sections
    assert interval > timestep_cm
    numsteps = int(interval // timestep_cm) + 2
    firstsection = True
    ti = 0
    initstate = initHEOMvec(rho_init)
    for s in range(sections):
        print('progress:', str(s+1), '/', str(sections))
        sec_tpoints = np.linspace(ti, ti+interval, num=numsteps)
        solution = solve_ivp(timederiv, (ti,ti+interval), initstate,\
                             t_eval=sec_tpoints)
        new_ts = solution.t[:-1]
        new_rhos = solution.y\
                        [HEOM_phys_ind*rhodim:(HEOM_phys_ind+1)*rhodim , :-1]
        if firstsection:
            tpoints = new_ts
            rhos = new_rhos
            firstsection = False
        else:
            tpoints = np.concatenate((tpoints, new_ts))
            rhos = np.concatenate((rhos, new_rhos), axis=1)
        ti += interval
        initstate = solution.y[:,-1]
    print('solve ODE time:', time.time()-starttime, 's')
    return tpoints * cm_to_fs, rhos
################################### Main ######################################
tpoints, rhos = solve_dynamics()
#tpoints, rhos = solve_dynamics_pw()
################################# Plotting ####################################
def plotrho(tpoints, rhopoints, row, col, form=None, part='R'):
    if part!='R' and part!='I':
        raise Exception('part needs to be either "R" or "I".')
    if part == 'R':
        ypoints = np.real(rhopoints[dim*row+col, :])
    else:
        ypoints = np.imag(rhopoints[dim*row+col, :])
    if form == None:
        plt.plot(tpoints, ypoints)
    else:
        plt.plot(tpoints, ypoints, form)
        
#np.save('FMO_7mer_lvl5_init6_pw_ts', tpoints)
#np.save('FMO_7mer_lvl5_init6_pw_rhos', rhos)
#plotrho(tpoints, rhos, 1, 1)
##plotrho(tpoints, rhos, 1, 1, part='I')
#plotrho(tpoints, rhos, 2, 2)
##plotrho(tpoints, rhos, 2, 2, part='I')
#plotrho(tpoints, rhos, 3, 3)
#plotrho(tpoints, rhos, 4, 4)
#plotrho(tpoints, rhos, 5, 5)
#plotrho(tpoints, rhos, 6, 6)
#plotrho(tpoints, rhos, 7, 7)
#plt.legend(['1','2','3','4','5','6','7'])
    
plt.plot(tpoints/1000, np.real(rhos[dim*0+0,:]), linestyle='dotted')


############################# monomer analytic ###############################
# from math import e
# def g(t):
#     s = t / cm_to_fs
#     temp = e**(-gamma*s)+gamma*s-1
#     return temp * (2*Lambda*kT_cm/gamma**2 -1j*Lambda/gamma)
# analytical_rho10 = np.array([e**(-g(t)) for t in tpoints])
# plt.plot(tpoints, np.imag(analytical_rho10))



