#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:12:22 2022

@author: LiwenKo
"""

from numpy import matmul, transpose, conjugate
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from math import pi, sqrt, e
import time
exciton_sys = 'dimer'
from parameters.dimer import H_sys, dipoles
import gc
import os
#from datetime import datetime

photon_n = 2
kT_K = 300 # in Kelvin
initsite = 0 # initial state's excited site. 0 for ground state.
#ctrfreq = 15222 # in invcm
ctrfreq = np.average(np.diag(H_sys)) # in invcm
polarization = np.array([0.4166278305478269, 0, -0.9090771423883736]) # unit vector in 3 spatial dimensions
tf_fs = 1000 # in fs
timestep_fs = tf_fs/1000 # in fs 
section = 50
obs_modes = [np.array([0.9090771423883736, 0, 0.4166278305478269]), np.array([0,1,0])]
geometric_factor_abs = 1
emission_type = "collective" # "None", "collective", or "independent"
piecewise = False
withflux = False
writeoutdata = False
foldername = 'data/dimer_1_photon_Fock_lvl5_03022022'
pulse_shape = 'Gaussian' # 'Gaussian', 'Square'
nus = np.array([-100, 200, 200])
gammas = np.array([0, 100, 100])
kappas = np.array([0, 50, 50])
HO_lvls = np.array([None, 5, 5])

kT = kT_K * 0.695 # in cm^-1
BE_nums = 1/(e**(nus/kT)-1)
cm_to_fs = 1e5/(6*pi) # the unit of time corresponding to 1 invcm, in fs
                      # or fs^-1 to cm^-1
sites = np.shape(H_sys)[0] # Number of sites
sysdim = sites + 1
sysrhodim = sysdim * sysdim
tf_cm = tf_fs/cm_to_fs # convert unit to cm
timestep_cm = timestep_fs/cm_to_fs
abs_tol = 1e-12
rel_tol = 1e-3
maxstep = tf_fs/20/cm_to_fs 
method = 'RK45'
n_water = 1.333
Gamma_factor = 1e5
WWgamma = 4/(3*1.054571)*(2*pi*ctrfreq)**3*1e-24\
         * n_water * (3*n_water**2/(2*n_water**2+1))**2 * Gamma_factor
            # Unit spont emission rate with omega = ctrfreq,
            # dipole = 1 debye. Final unit is in fs^-1.#WWgamma = 0.1/16 # in fs^-1
WWgamma = WWgamma * cm_to_fs # convert unit to cm^-1

bandwidth_fs = 0.06 #1*0.056381983409348496 # in fs^-1
bandwidth = bandwidth_fs * cm_to_fs # convert unit to cm^-1
#bandwidth = sqrt(np.var(np.diag(H_sys))) # in cm^-1
#bandwidth_fs = bandwidth / cm_to_fs

pulse_offset_fs = 200 # (fs)
pulse_offset = pulse_offset_fs / cm_to_fs
pulse_cutoff = pulse_offset + 15/bandwidth

################################# Temporal profile ############################
if pulse_shape == 'Gaussian':
    def timeprof(t):
        ''' Time profile of the pulse in units of invcm^-1 '''
        return (bandwidth**2/(2*pi))**(1/4) * \
                e**(-bandwidth**2*(t-pulse_offset)**2/4)
elif pulse_shape == 'Square':
    def timeprof(t):
        if np.absolute(t-pulse_offset) <= 1/(2*bandwidth):
            return bandwidth
        else:
            return 0
#################### Import Hamiltonian, set up initial state #################
def prod(*args):
    rtn = 1
    for arg in args:
        rtn = np.kron(rtn, arg)
    return rtn
def lowering(HO_lvl):
    '''returns the lowering operator in the truncated HO_lvl-dimensional 
    harmonic oscillator space. '''
    rtn = np.zeros((HO_lvl, HO_lvl), dtype=complex)
    for l in range(HO_lvl):
        if l != 0:
            rtn[l-1, l] = sqrt(l)
    return rtn
def raising(HO_lvl):
    '''returns the lowering operator in the truncated HO_lvl-dimensional 
    harmonic oscillator space. '''
    rtn = np.zeros((HO_lvl, HO_lvl), dtype=complex)
    for l in range(HO_lvl):
        if l != HO_lvl-1:
            rtn[l+1, l] = sqrt(l+1)
    return rtn
def position(HO_lvl):
    '''returns the position operator.'''
    return (lowering(HO_lvl) + raising(HO_lvl))/sqrt(2)
def eye(dim):
    return np.identity(dim, dtype=complex)
def H_HO(HO_lvl, nu):
    '''returns the harmonic oscillator Hamiltonain with HO_lvl truncated
    levels. nu: frequency of HO.'''
    rtn = np.zeros((HO_lvl, HO_lvl), dtype=complex)
    for l in range(HO_lvl):
        rtn[l, l] = nu * l
    return rtn
def thermal_state(HO_lvl, BEnum):
    '''returns the truncated thermal state density matrix with Bose-Einstein
    number = BEnum.'''
    Z_HO = sum([(BEnum/(BEnum+1))**i for i in range(HO_lvl)])
    rtn = np.zeros((HO_lvl, HO_lvl), dtype=complex)
    for l in range(HO_lvl):
        rtn[l, l] = ((BEnum/(BEnum+1)) ** l) / Z_HO
    return rtn

print('Initializing...')
# ground state is indexed 0
H_sites = H_sys - np.identity(sites) * ctrfreq
sysH = np.zeros((sysdim, sysdim), dtype=complex)
sysH[1:, 1:] = H_sites
sysrho_init = np.zeros((sysdim, sysdim), dtype=complex)
sysrho_init[initsite, initsite] = 1
def projection(site):
    '''returns the projection operator to single exciton state at site Site'''
    temp = np.zeros([sysdim, sysdim], dtype=complex)
    temp[site, site] = 1
    return temp
projs = [None]
for i in range(sites):
    projs.append(projection(i+1))

totdim = int(sysdim * np.prod(HO_lvls[1:]))
rhodim = totdim**2
totalH = np.zeros((totdim, totdim), dtype=complex)
totalH += prod(sysH, eye(HO_lvls[1]), eye(HO_lvls[2]))
totalH += prod(eye(sysdim), H_HO(HO_lvls[1],nus[1]), eye(HO_lvls[2]))
totalH += prod(eye(sysdim), eye(HO_lvls[1]), H_HO(HO_lvls[2],nus[2]))
totalH += kappas[1]*prod(projs[1], position(HO_lvls[1]), eye(HO_lvls[2]))
totalH += kappas[2]*prod(projs[2], eye(HO_lvls[1]), position(HO_lvls[2]))

total_rho_init = prod(sysrho_init, thermal_state(HO_lvls[1], BE_nums[1]), \
                      thermal_state(HO_lvls[2], BE_nums[2]))
#total_rho_init_flat = np.reshape(total_rho_init, -1)
########################## Absorption and emission L's ########################
indiv_Ls = {}
for i in range(1, sites+1):
    temp = np.zeros((sysdim, sysdim), dtype=complex)
    temp[0,i] = 1
    temp = prod(temp, eye(HO_lvls[1]), eye(HO_lvls[2]))
    indiv_Ls[i] = sqrt(WWgamma)*temp
def collectiveL(vec):
    ''' Returns a linear combination of individual L's, 
    weighted according to vec. '''
    if np.isscalar(vec): # monomer special case
        return collectiveL([vec])
    rtn = np.zeros((totdim, totdim), dtype=complex)
    for i in range(1, sites+1):
        rtn += vec[i-1] * indiv_Ls[i]
    return rtn
    
L_abs = collectiveL(matmul(dipoles, polarization))*sqrt(geometric_factor_abs)
L_emi = []
if emission_type == "None":
    None
elif emission_type == "collective":
    for i in range(3): # collective emission
        L_emi.append(collectiveL(dipoles[:,i]))
elif emission_type == "independent":
    for i in range(1, sites+1):  # independent emission
        L_emi.append(indiv_Ls[i]*np.linalg.norm(dipoles[i-1]))
else:
    raise Exception("undefined emission type")
########################## phonon raising and lowering ########################
raising1 = prod(eye(sysdim), raising(HO_lvls[1]), eye(HO_lvls[2]))
lowering1 = prod(eye(sysdim), lowering(HO_lvls[1]), eye(HO_lvls[2]))
raising2 = prod(eye(sysdim), eye(HO_lvls[1]), raising(HO_lvls[2]))
lowering2 = prod(eye(sysdim), eye(HO_lvls[1]), lowering(HO_lvls[2]))

######################## FS Indexing ################################

FS_lvl_vecs = []
for m in range(photon_n+1):
    for n in range(photon_n+1):
        FS_lvl_vecs.append((m, n))

FS_lvl_ind = {}
for i in range(len(FS_lvl_vecs)):
    FS_lvl_ind[FS_lvl_vecs[i]] = i

totlvls = len(FS_lvl_vecs)
physvec = (photon_n, photon_n)
FSlvls = int((photon_n + 1) * (photon_n + 1))
physical_ind = FS_lvl_ind[physvec]

########################### FS Indexing functions ###################
def getlevelvector(levelindex):
    return FS_lvl_vecs[levelindex]
def getlevelindex(levelvector):
    return FS_lvl_ind[levelvector]
def getFSlowerrow(vec):
    temp = np.array(vec)
    if temp[-2] == 0:
        return None
    else:
        temp[-2] -= 1
        return tuple(temp)
def getFSlowercol(vec):
    temp = np.array(vec)
    if temp[-1] == 0:
        return None
    else:
        temp[-1] -= 1
        return tuple(temp)  
def getFShigherrow(vec):
    temp = np.array(vec)
    if temp[-2] == photon_n:
        return None
    else:
        temp[-2] += 1
        return tuple(temp)
def getFShighercol(vec):
    temp = np.array(vec)
    if temp[-1] == photon_n:
        return None
    else:
        temp[-1] += 1
        return tuple(temp)
def FSrow(vec):
    return vec[-2]
def FScol(vec):
    return vec[-1]
def equalFSrowcol(vec):
    return vec[-2] == vec[-1]

########################## Useful functions and variables #####################

def commutator(A, B):
    return matmul(A, B) - matmul(B, A)
def anticommutator(A, B):
    return matmul(A, B) + matmul(B, A)
def dagger(A):
    return transpose(conjugate(A))
def dissipator(L, rho):
    temp1 = matmul(L, matmul(rho, dagger(L)))
    temp2 = 1/2 * matmul(dagger(L), matmul(L, rho))
    temp3 = 1/2 * matmul(rho, matmul(dagger(L), L))
    return temp1 - temp2 - temp3
########## use superoperators on vectorized rhos to reduce runtime ###########
#### instead of reshaping vectorized and matrix rho back and forth
#eye = np.identity(dim, dtype=complex)
#eye_super = np.kron(eye, eye)
#def leftmult_super(L):
#    return np.kron(L, eye)
#def rightmult_super(L):
#    return np.kron(eye, transpose(L))
#def commut_super(L):
#    return leftmult_super(L) - rightmult_super(L)
#def anticommut_super(L):
#    return leftmult_super(L) + rightmult_super(L)
#def dissipator_super(L):
#    return matmul(leftmult_super(L), rightmult_super(dagger(L))) \
#        - 1/2 * leftmult_super(matmul(dagger(L), L)) \
#        - 1/2 * rightmult_super(matmul(dagger(L), L))
#def dagger_super(vecrho):
#    return np.reshape(dagger(np.reshape(vecrho, (dim,dim))), -1)
#H_commut_super = commut_super(H)
#total_diss_super = np.zeros((dim**2, dim**2), dtype=complex)
#for L in L_emi:
#    total_diss_super += dissipator_super(L)
#L_abs_commut_super = commut_super(L_abs)
#L_abs_dag_commut_super = commut_super(dagger(L_abs))
#proj_commut_supers = [None]
##proj_anticommut_supers = [None]
#proj_leftmult_supers = [None]
#proj_rightmult_supers = [None]
#for i in range(1, sites+1):
#    proj_commut_supers.append(commut_super(projs[i]))
##    proj_anticommut_supers.append(anticommut_super(projs[i]))
#    proj_leftmult_supers.append(leftmult_super(projs[i]))
#    proj_rightmult_supers.append(rightmult_super(projs[i]))
## a vector v s.t. Tr(vec(x)) = v dot vec(x)
#tr_super = np.zeros(rhodim, dtype=complex) 
#for i in range(dim):
#    tr_super[i*dim+i] = 1
############ Methods for accessing rhos in vectorized FS+HEOM state ###########
def getrho(state, lvlvec):
    ''' Returns the auxiliary rho (matrix) with index lvlvec 
    from vectorized FS+HEOM state. '''
    assert len(lvlvec) == 2
    temp = FS_lvl_ind[tuple(lvlvec)]
    return np.reshape(state[temp*rhodim: (temp+1)*rhodim], (totdim,totdim))
def addrho(state, lvlvec, rho):
    ''' Add rho (matrix) to the auxiliary rho indexed lvlvec.
    rho is in matrix form with dimension (totdim * totdim)'''
    assert len(lvlvec) == 2
    temp2 = FS_lvl_ind[tuple(lvlvec)]
    state[temp2*rhodim: (temp2+1)*rhodim] += np.reshape(rho, -1)
    return None
def initFSvec(rhoinit):
    ''' Returns the vectorized FS state (with all the auxiliary rhos) '''
    rtn = np.zeros(totlvls*rhodim, dtype=complex)
    temp = list(physvec)
    rhovec = np.reshape(rhoinit, -1)
    for i in range(photon_n+1):
        addrho(rtn, temp, rhovec)
        if FScol(temp) > 0:
            temp = getFSlowerrow(getFSlowercol(temp))
    return rtn
############################## timederiv function #####################
def timederiv_withpulse(t, state):
    '''The time derivative function during pulse. This acts on FS+HEOM vectors.
    Include both photon and phonon terms.'''
    rtn_state = np.zeros(totlvls*rhodim, dtype=complex)
    for v in FS_lvl_vecs:
        temp = np.zeros((totdim, totdim), dtype=complex)
        rhoin = getrho(state, v)
        temp += -1j * commutator(totalH, rhoin)
        for L in L_emi:
            temp += dissipator(L, rhoin)
        temp += gammas[1] * BE_nums[1] * dissipator(raising1, rhoin)
        temp += gammas[1] * (BE_nums[1]+1) * dissipator(lowering1, rhoin)
        temp += gammas[2] * BE_nums[2] * dissipator(raising2, rhoin)
        temp += gammas[2] * (BE_nums[2]+1) * dissipator(lowering2, rhoin)
        
        lowerFSrowvec = getFSlowerrow(v)
        lowerFScolvec = getFSlowercol(v)
        if lowerFSrowvec != None:
            rhoin = getrho(state, lowerFSrowvec)
            temp = temp - sqrt(FSrow(v)) * timeprof(t) * \
                    commutator(dagger(L_abs), rhoin)
        if lowerFScolvec != None:
            rhoin = getrho(state, lowerFScolvec)
            temp = temp + sqrt(FScol(v)) * timeprof(t).conjugate() * \
                    commutator(L_abs, rhoin)
        addrho(rtn_state, v, temp)
    return rtn_state

def timederiv_nopulse(t, state):
    '''The time derivative function after the pulse. 
    This acts on HEOM vectors (i.e. reduced from FS+HEOM vectors).'''
    rtn_state = np.zeros(totlvls*rhodim, dtype=complex)
    for v in FS_lvl_vecs:
        temp = np.zeros((totdim, totdim), dtype=complex)
        rhoin = getrho(state, v)
        temp += -1j * commutator(totalH, rhoin)
        for L in L_emi:
            temp += dissipator(L, rhoin)
        temp += gammas[1] * BE_nums[1] * dissipator(raising1, rhoin)
        temp += gammas[1] * (BE_nums[1]+1) * dissipator(lowering1, rhoin)
        temp += gammas[2] * BE_nums[2] * dissipator(raising2, rhoin)
        temp += gammas[2] * (BE_nums[2]+1) * dissipator(lowering2, rhoin)
        
        addrho(rtn_state, v, temp)
    return rtn_state
##################################### flux ####################################
#L_obs = []
#for m in obs_modes:
#    L_obs.append(collectiveL(matmul(dipoles, m)))
#trL_supers = []
#trLdL_supers = []
#for L in L_obs:
#    trL_supers.append(matmul(tr_super, leftmult_super(L)))
#    trLdL_supers.append(matmul(tr_super, matmul(leftmult_super(dagger(L)),\
#                                                leftmult_super(L))))
#trL_abs_super = matmul(tr_super, leftmult_super(L_abs))
#trLdL_abs_super = matmul(tr_super, matmul(leftmult_super(dagger(L_abs)),\
#                                          leftmult_super(L_abs)))
#def transflux(t, FSHEOMvec, reduced=False):
#    ''' Returns the transmitted photon flux in units of ns^-1. 
#    t is time in cm^-1. '''
#    if reduced:
#        rtn = matmul(trLdL_abs_super, getrho_reduced(FSHEOMvec, reducevec(physvec)))
#        return rtn / cm_to_fs * 1e6
#    rtn = photon_n * abs(timeprof(t))**2
#    temp = sqrt(photon_n) * timeprof(t).conjugate() * \
#        matmul(trL_abs_super, getrho(FSHEOMvec, getFSlowercol(physvec)))
#    rtn += 2*temp.real
#    rtn += matmul(trLdL_abs_super, getrho(FSHEOMvec, physvec))
#    return rtn / cm_to_fs * 1e6
##    return 2*temp.real / cm_to_fs
##    return matmul(trLdL_abs_super, getrho(FSHEOMvec, physvec))
##    return photon_n * abs(timeprof(t))**2
#    
#def scatflux(statevec, reduced=False):
#    ''' Returns the scattered photon flux (ns^-1) as a list, whose elements
#    are indexed the same way as obs_modes. '''
#    rtn = np.zeros(len(obs_modes), dtype=complex)
#    for i in range(len(obs_modes)):
#        if reduced:
#            rtn[i] = matmul(trLdL_supers[i], getrho_reduced(statevec, reducevec(physvec)))
#        else:
#            rtn[i] = matmul(trLdL_supers[i], getrho(statevec, physvec))
#    return rtn / cm_to_fs * 1e6
################################## integrate ODE ##############################

def solvedynamics_full(flux=False):
    '''Solve dynamics while keeping all FS+HEOM rhos for all time points. '''
    starttime = time.time()
    initstate = initFSvec(total_rho_init)
    tpoints = np.arange(0, tf_cm, timestep_cm)   
    solution = solve_ivp(timederiv_withpulse, (0,tf_cm), initstate, \
                         t_eval=tpoints, atol=abs_tol, rtol=rel_tol, max_step=maxstep, method=method)
#    solution = solve_ivp(timederiv_withpulse, (0,tf_cm), initstate, \
#                         atol=abs_tol, method=method)
    tpoints = solution.t
    rhos = solution.y[physical_ind*rhodim:(physical_ind+1)*rhodim,:]
#    rhos = solution.y[1*rhodim:(1+1)*rhodim,:]
    finalstate = solution.y[:,-1]
    print('solveivp time:', str(time.time()-starttime), 's')
    if flux:
        transfluxes = np.zeros(len(tpoints), dtype=complex)
        for i in range(len(tpoints)):
            transfluxes[i] = transflux(tpoints[i], solution.y[:,i])
        scatfluxes = np.zeros((len(obs_modes), len(tpoints)), dtype=complex)     
        for i in range(len(tpoints)):
            scatfluxes[:,i] = scatflux(solution.y[:,i])
        return tpoints*cm_to_fs, rhos, finalstate, transfluxes, scatfluxes   
    else:
        return tpoints*cm_to_fs, rhos, finalstate

def solvedynamics_cutoff_pw(flux=False):
    '''Solve dynamics with pulse cutoff. Integrate ODE piecewise to reduce
    momory cost associated with unused auxiliary rhos.'''
    print('solving ODE piecewise...')
    starttime = time.time()   
    interval = tf_cm/section
    assert interval > timestep_cm
    pulsesections = int(pulse_cutoff//interval) + 1
    numsteps = int(interval // timestep_cm) + 2
    firstsection = True
    ti = 0
    initstate = initFSHEOMvec(rho_init)
    if tf_cm <= pulse_cutoff:
        print('warning: untested code')
        for s in range(section):
            print('progress:', str(s+1), '/', str(section))
            sec_tpoints = np.linspace(ti, ti+interval, num=numsteps)
            solution = solve_ivp(timederiv_withpulse, (ti,ti+interval), initstate,\
                                 t_eval=sec_tpoints, atol=abs_tol, rtol=rel_tol, max_step=maxstep, method=method)
            new_ts = np.copy(solution.t[:-1])
            new_rhos = np.copy(solution.y\
                            [physical_ind*rhodim:(physical_ind+1)*rhodim , :-1])
            if flux:
                new_transfluxes = np.zeros(len(new_ts), dtype=complex)
                new_scatfluxes = np.zeros((len(obs_modes),len(new_ts)), dtype=complex)
                for i in range(len(new_ts)):
                    new_transfluxes[i] = transflux(new_ts[i], solution.y[:,i])
                    new_scatfluxes[:,i] = scatflux(solution.y[:,i])
            if firstsection:
                tpoints = new_ts
                rhos = new_rhos
                if flux:
                    transfluxes = new_transfluxes
                    scatfluxes = new_scatfluxes
                firstsection = False
            else:
                tpoints = np.concatenate((tpoints, new_ts))
                rhos = np.concatenate((rhos, new_rhos), axis=1)
                if flux:
                    transfluxes = np.concatenate((transfluxes, new_transfluxes))
                    scatfluxes = np.concatenate((scatfluxes, new_scatfluxes), axis=1)
            ti += interval
            initstate = solution.y[:,-1]
            if s == section - 1:
                finalstate = np.copy(solution.y[:, -1])
            del solution
            gc.collect()
        print('solveivp time:', str(time.time()-starttime), 's')
        if flux:
            return tpoints*cm_to_fs, rhos, finalstate, transfluxes, scatfluxes
        return tpoints*cm_to_fs, rhos, finalstate

    for s in range(pulsesections):
        print('progress:', str(s+1), '/', str(section))
        sec_tpoints = np.linspace(ti, ti+interval, num=numsteps)
        solution = solve_ivp(timederiv_withpulse, (ti,ti+interval), initstate,\
                             t_eval=sec_tpoints, atol=abs_tol, rtol=rel_tol, max_step=maxstep, method=method)
        new_ts = np.copy(solution.t[:-1])
        new_rhos = np.copy(solution.y\
                        [physical_ind*rhodim:(physical_ind+1)*rhodim , :-1])
        if flux:
            new_transfluxes = np.zeros(len(new_ts), dtype=complex)
            new_scatfluxes = np.zeros((len(obs_modes),len(new_ts)), dtype=complex)
            for i in range(len(new_ts)):
                new_transfluxes[i] = transflux(new_ts[i], solution.y[:,i])
                new_scatfluxes[:,i] = scatflux(solution.y[:,i])
        if firstsection:
            tpoints = new_ts
            rhos = new_rhos
            if flux:
                transfluxes = new_transfluxes
                scatfluxes = new_scatfluxes
            firstsection = False
        else:
            tpoints = np.concatenate((tpoints, new_ts))
            rhos = np.concatenate((rhos, new_rhos), axis=1)
            if flux:
                transfluxes = np.concatenate((transfluxes, new_transfluxes))
                scatfluxes = np.concatenate((scatfluxes, new_scatfluxes), axis=1)
        ti += interval
        initstate = solution.y[:,-1]
        del solution
        gc.collect()
    # Reduce FS+HEOM vec to HEOM vec
    print('Pulse cut off. Reduce FS+HEOM to HEOM.')
    initstate = reduceFSHEOM(initstate)
    for s in range(section - pulsesections):
        print('progress:', str(s+pulsesections+1), '/', str(section))
        sec_tpoints = np.linspace(ti, ti+interval, num=numsteps)
        solution = solve_ivp(timederiv_nopulse, (ti,ti+interval), initstate,\
                             t_eval=sec_tpoints, atol=abs_tol, rtol=rel_tol, max_step=maxstep, method=method)
        new_ts = solution.t[:-1]
        new_rhos = np.copy(solution.y\
                [HEOM_physical_ind*rhodim:(HEOM_physical_ind+1)*rhodim , :-1])
        if flux:
            new_transfluxes = np.zeros(len(new_ts), dtype=complex)
            new_scatfluxes = np.zeros((len(obs_modes),len(new_ts)), dtype=complex)
            for i in range(len(new_ts)):
                new_transfluxes[i] = transflux(new_ts[i], solution.y[:,i], reduced=True)
                new_scatfluxes[:,i] = scatflux(solution.y[:,i], reduced=True)
            transfluxes = np.concatenate((transfluxes, new_transfluxes))
            scatfluxes = np.concatenate((scatfluxes, new_scatfluxes), axis=1)
        tpoints = np.concatenate((tpoints, new_ts))
        rhos = np.concatenate((rhos, new_rhos), axis=1)
        ti += interval
        initstate = solution.y[:,-1]
        if s == section - pulsesections -1:
            finalstate = np.copy(solution.y[:, -1])
        del solution
        gc.collect()
    print('solveivp time:', str(time.time()-starttime), 's')
    if flux:
        return tpoints*cm_to_fs, rhos, finalstate, transfluxes, scatfluxes
    return tpoints*cm_to_fs, rhos, finalstate

################################# Plotting ####################################
def Trvib(rho):
    rtn = np.zeros((sysdim, sysdim), dtype=complex)
    for i in range(sysdim):
        for j in range(sysdim):
            temp = np.prod(HO_lvls[1:])
            rtn[i,j] = np.trace(rho[i*temp:(i+1)*temp, j*temp:(j+1)*temp])
    return rtn

#rhos_e = np.zeros((sites, sites, len(tpoints)), dtype=complex)
#for i in range(len(tpoints)):
#    temp = np.reshape(solution_e.y[:,i], (sites*HO_lvl, sites*HO_lvl))
#    rhos_e[:,:,i] = Trvib(temp)
def plotrho(tpoints, rhopoints, row, col, form=None, part='R', yscale=1):
    if part!='R' and part!='I':
        raise Exception('part needs to be either "R" or "I".')
    if part == 'R':
        ypoints = np.real(rhopoints[sysdim*row+col, :])
    else:
        ypoints = np.imag(rhopoints[sysdim*row+col, :])
    if form == None:
        plt.plot(tpoints, ypoints*yscale)
    else:
        plt.plot(tpoints, ypoints*yscale, form)
#    plt.ylabel('x $10^{-6}$')
    plt.xlabel('time (fs)')
    
def plotflux(tpoints, fluxes, mode, part='R'):
    '''Plot photon flux. mode = 'trans' or 'scat' for transmitted or scattered.
    If mode='trans', then fluxes is an 1-d array. If mode='scat', then fluxes
    is a (number of scattered mode) x (number of tpoints) 2-d array. '''
    if mode == 'trans':
        assert len(np.shape(fluxes)) == 1
        if part == 'R':
            plt.plot(tpoints, np.real(fluxes))
        elif part == 'I':
            plt.plot(tpoints, np.imag(fluxes))
    elif mode == 'scat':
        assert np.shape(fluxes)[0] == len(obs_modes)
        for i in range(len(obs_modes)):
            if part == 'R':
                plt.plot(tpoints, np.real(fluxes[i,:]))
            elif part == 'I':
                plt.plot(tpoints, np.imag(fluxes[i,:]))
    else:
        raise Exception('unrecognized mode')
    
def plotpulse(scale=0.2):
    tpoints = np.arange(0, tf_cm, timestep_cm)
#    pulsepts = timeprof(tpoints)
    pulsepts = np.array([timeprof(t) for t in tpoints])
    plt.plot(tpoints * cm_to_fs, pulsepts*scale)
#    plt.fill_between((tpoints * cm_to_fs)[50:250], (pulsepts*scale)[50:250], color='lightgray')
########################## Display info and decay rate ########################
def displayparam():
    print('HEOM level =', str(lvl))
    print('photon number =', str(photon_n))
    print('sites =', str(sites))
    print('number of Fock+HEOM levels =', str(totlvls))
    print('final time =', str(tf_fs), 'fs')
    print('timestep =', str(timestep_fs), 'fs')
def displaysteadystateinfo(steadystate):
    '''Returns the steady state decay rate in units of ns^-1 '''
    temprho = np.reshape(steadystate, (dim, dim))[1:, 1:]
    excitedpop = np.trace(temprho).real
    print('Steady excited state population:', str(excitedpop))
    temprho = temprho/np.trace(temprho)
    temprho2 = np.zeros((dim,dim), dtype=complex)
    temprho2[1:, 1:] = temprho
    rate = 0
    for L in L_emi:
        LdL = matmul(dagger(L), L)
        rate += np.trace(matmul(LdL, temprho2))
    rate = (rate/cm_to_fs).real * 1e6
    print('Steady state decay rate:', str(rate), 'ns^-1')
    print('Steady state total photon flux:', str(excitedpop*rate), 'ns^-1')
############################### Main ##########################################
if writeoutdata:
#    foldername = 'data/'+datetime.today().strftime('%Y%m%d%H%M')
    os.mkdir(foldername)
    logfile = open(foldername+'/log.txt', 'w')
    logfile.write('Fock + HEOM calculation\n\n')
    logfile.write('General parameters \n')
    logfile.write('Exciton system: ' + exciton_sys + '\n')
    logfile.write('number of sites: ' + str(sites) + '\n')
#    logfile.write('HEOM lvl: ' + str(lvl) + '\n')
    logfile.write('Temperature: ' + str(kT_K) + ' Kelvin \n')
    logfile.write('Photon number: ' + str(photon_n) + '\n')
    logfile.write('Total number of Fock+HEOM levels: ' + str(totlvls) + '\n')
    logfile.write('Pulse center frequency: ' + str(ctrfreq) + ' cm^-1 \n')
    logfile.write('Pulse center time: ' + str(pulse_offset_fs) + ' fs \n')
    logfile.write('Pulse bandwidth: ' + str(bandwidth_fs) + ' fs^-1 \n')
    logfile.write('Absorption mode geometric factor: ' + str(geometric_factor_abs) + '\n')
    logfile.write('Pulse polarization: ' + str(polarization) + '\n')
    logfile.write('Other polarization modes: ' + str(np.array(obs_modes).tolist()) + '\n\n')
    logfile.write('Integration parameters \n')
    logfile.write('Final time: ' + str(tf_fs) + ' fs \n')
    logfile.write('Absolute tolerance: ' + str(abs_tol) + '\n')
    logfile.write('Sampled timestep: ' + str(timestep_fs) + ' fs \n')
    logfile.write('Piecewise integration with pulse cutoff: ' + str(piecewise) + '\n')
#    logfile.write('Modified HEOM terminator EOM: ' + str(special_terminators) + '\n')
    logfile.write('Index of refraction (water): ' + str(n_water) + '\n')
    logfile.write('Gamma blow-up factor: ' + str(Gamma_factor) + '\n')
    logfile.write('Emission type: ' + emission_type + '\n')
    if piecewise:
        logfile.write('Number of piecewise sections: ' + str(section) + '\n')
    logfile.write('Flux Calculation: ' + str(withflux) + '\n')
    displayparam()
    print('Solving ODE...')
    starttime = time.time()
if withflux:
    if piecewise:
        tpoints, rhopoints, finalstate, transfluxes, scatfluxes = solvedynamics_cutoff_pw(flux=True)
    else:
        tpoints, rhopoints, finalstate, transfluxes, scatfluxes = solvedynamics_full(flux=True) # slow
    if writeoutdata:
        logfile.write('ODE integration time: ' + str(time.time()-starttime) + ' s \n')
        np.save(foldername+'/rhopoints', rhopoints)
        np.save(foldername+'/tpoints', tpoints)
        np.save(foldername+'/finalstate', finalstate)
        np.save(foldername+'/transfluxes', transfluxes)
        np.save(foldername+'/scatfluxes', scatfluxes)
else:
    if piecewise:
        tpoints, rhopoints, finalstate = solvedynamics_cutoff_pw()
    else:
        tpoints, rhopoints, finalstate = solvedynamics_full() # slow
    if writeoutdata:
        logfile.write('ODE integration time: ' + str(time.time()-starttime) + ' s \n')
        np.save(foldername+'/rhopoints', rhopoints)
        np.save(foldername+'/tpoints', tpoints)
        np.save(foldername+'/finalstate', finalstate)
if writeoutdata:
    logfile.close()

## plotting
reduced_rhopoints = np.zeros((sysdim,sysdim,len(tpoints)), dtype=complex)
for i in range(len(tpoints)):
    temp = np.reshape(rhopoints[:,i], (totdim, totdim))
    reduced_rhopoints[:,:,i] = Trvib(temp)
plt.plot(tpoints, np.real(reduced_rhopoints[1,1,:]), linestyle = 'dotted', linewidth=3)
plt.plot(tpoints, np.real(reduced_rhopoints[2,2,:]), linestyle = 'dotted', linewidth=3)

#plotrho(tpoints, reduced_rhopoints, 1, 1)
#plotrho(tpoints, reduced_rhopoints, 2, 2)
#plotrho(tpoints, rhopoints, 1, 2, part='R')
#plotrho(tpoints, rhopoints, 1, 2, part='I')

#temp = rhopoints[4,:] + rhopoints[8,:]
#plt.plot(tpoints, np.real(temp))
#eigvals, eigvecs = np.linalg.eig(H)
#rhopoints_eig = np.zeros_like(rhopoints, dtype=complex)
#for i in range(len(tpoints)):
#    temp = np.reshape(rhopoints[:,i], (dim,dim))
#    rhopoints_eig[:,i] = np.reshape(matmul(dagger(eigvecs), matmul(temp, eigvecs)), -1)
#plotrho(tpoints, rhopoints_eig, 2, 2)
#plotrho(tpoints, rhopoints_eig, 1, 1)
#plotflux(tpoints, scatfluxes, 'scat')
#fig = plt.figure()
#from mpl_toolkits.mplot3d import Axes3D
#ax = Axes3D(fig)
#final_state = rhopoints[:, -1]
#_x = np.arange(dim)
#_y = np.arange(dim)
#_xx, _yy = np.meshgrid(_x, _y)
#x, y = _xx.ravel(), _yy.ravel()
#top = np.absolute(final_state)
#bottom = np.zeros_like(top)
#ax.bar3d(x, y, bottom, 1, 1, top)

#overall_WWgamma = np.linalg.norm(matmul(dipoles, polarization))**2*WWgamma
#print('Gamma: '+ str(overall_WWgamma))
#abs_prob = np.real(np.trace(np.reshape(rhopoints[:, -1], (dim,dim))[1:, 1:]))
#print('abs prob: ' + str(abs_prob))
#print('ratio: ' + str(abs_prob/overall_WWgamma))
#totprob = rhopoints[1*dim+1,:]+ rhopoints[2*dim+2,:]
#plt.plot(tpoints, np.real(totprob))
#print('end abs prob:', str(totprob[-1]))