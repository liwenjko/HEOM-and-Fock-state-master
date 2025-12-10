#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:18:46 2021

@author: LiwenKo
"""

from functools import lru_cache
from numpy import matmul, transpose, conjugate
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from math import pi, sqrt, e
import time
exciton_sys = 'dimer'
from parameters.dimer import H_sys, dipoles, lambdas, gammas
import gc
import os
#from datetime import datetime

lvl = 5
kT_K = 300 # in Kelvin
initsite = 0 # initial state's excited site. 0 for ground state.
ctrfreq = 15222 # in invcm
#ctrfreq = np.average(np.diag(H_sys)) # in invcm
polarization = np.array([0.4166278305478269, 0, -0.9090771423883736]) # unit vector in 3 spatial dimensions
tf_fs = 1000 # in fs
timestep_fs = 1 # in fs 
section = 200 
obs_modes = [np.array([0,1,0]), np.array([0.9090771423883736, 0, 0.4166278305478269])]
geometric_factor_abs = 1
cut_off_pulse = False
piecewise = False
withflux = True
writeoutdata = True
foldername = 'data/dimer_1_photon_coherent_lvl5_03022022'
special_terminators = False # Turn to False if no phonon

kT = kT_K * 0.695 # in cm^-1
cm_to_fs = 1e5/(6*pi) # the unit of time corresponding to 1 invcm, in fs
                      # or fs^-1 to cm^-1
sites = np.shape(H_sys)[0] # Number of sites
dim = sites + 1
rhodim = dim * dim
tf_cm = tf_fs/cm_to_fs # convert unit to cm
timestep_cm = timestep_fs/cm_to_fs
abs_tol = 1e-12
rel_tol = 1e-6
maxstep = 100/cm_to_fs
method = 'RK45'
n_water = 1.333
Gamma_factor = 1
WWgamma = 4/(3*1.054571)*(2*pi*ctrfreq)**3*1e-24\
         * n_water * (3*n_water**2/(2*n_water**2+1))**2 * Gamma_factor
            # Unit spont emission rate with omega = ctrfreq,
            # dipole = 1 debye. Final unit is in fs^-1.#WWgamma = 0.1/16 # in fs^-1
WWgamma = WWgamma * cm_to_fs # convert unit to cm^-1

bandwidth_fs = 0.06 #10*0.0611 #1*0.056381983409348496 # in fs^-1
bandwidth = bandwidth_fs * cm_to_fs # convert unit to cm^-1
#bandwidth = sqrt(np.var(np.diag(H_sys))) # in cm^-1
#bandwidth_fs = bandwidth / cm_to_fs

pulse_offset_fs = 200 # (fs)
pulse_offset = pulse_offset_fs / cm_to_fs
pulse_cutoff = pulse_offset + 10/bandwidth
lambdas = np.insert(lambdas, 0, 0)
gammas = np.insert(gammas, 0, 0)
alpha = sqrt(1) # coherent state amplitude. |alpha|^2 = average photon number

################################# Temporal profile ############################
def timeprof(t):
    ''' Time profile of the pulse in units of invcm^-1 '''
    return alpha * (bandwidth**2/(2*pi))**(1/4) * \
            e**(-bandwidth**2*(t-pulse_offset)**2/4)

#################### Import Hamiltonian, set up initial state #################
print('Initializing...')
# ground state is indexed 0
H_sites = H_sys - np.identity(sites) * ctrfreq
H = np.zeros((dim, dim), dtype=complex)
H[1:, 1:] = H_sites
rho_init = np.zeros((dim, dim), dtype=complex)
rho_init[initsite, initsite] = 1
assert np.shape(H) == (dim, dim)
########################## Absorption and emission L's ########################
indiv_Ls = {}
for i in range(1, sites+1):
    temp = np.zeros((dim,dim), dtype=complex)
    temp[0,i] = 1
    indiv_Ls[i] = sqrt(WWgamma)*temp
def collectiveL(vec):
    ''' Returns a linear combination of individual L's, 
    weighted according to vec. '''
    if np.isscalar(vec): # monomer special case
        return collectiveL([vec])
    rtn = np.zeros((dim, dim), dtype=complex)
    for i in range(1, sites+1):
        rtn += vec[i-1] * indiv_Ls[i]
    return rtn
    
L_abs = collectiveL(matmul(dipoles, polarization))*sqrt(geometric_factor_abs)
L_emi = []
for i in range(3): # collective emission
    L_emi.append(collectiveL(dipoles[:,i]))
#for i in range(1, sites+1):  # independent emission
#    L_emi.append(indiv_Ls[i]*np.linalg.norm(dipoles[i-1]))

######################## FS+HEOM/HEOM Indexing ################################
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
# First two indices are for FS master equations. Last two indices are for HEOM.
# Only keep the FS rhos where row > col
for v in range(lvl+1):
    templst = construct_HEOM_lvl_lst(v, sites)
    HEOM_lvl_vecs.extend(templst)

HEOM_lvl_ind = {}
for i in range(len(HEOM_lvl_vecs)):
    temp = tuple(HEOM_lvl_vecs[i])
    HEOM_lvl_vecs[i] = temp
    HEOM_lvl_ind[temp] = i
HEOMlvls = len(HEOM_lvl_vecs)
physvec = np.zeros(sites, dtype=int)
physvec = tuple(physvec)
physical_ind = HEOM_lvl_ind[physvec]

########################### HEOM Indexing functions ###################
def getlevelvector(levelindex):
    return HEOM_lvl_vecs[levelindex]
def getlevelindex(levelvector):
    return HEOM_lvl_ind[levelvector]
def getHEOMhighervec(vec, site):
    ''' Returns the vector with site having one higher level. Returns None if
    the higher vector is beyond the HEOM level. '''
    temp = np.array(vec)
    temp[site-1] += 1
    temp2 = tuple(temp)
    if sumHEOMsitelvl(temp2) <= lvl:
        return temp2
    else:
        return None
def getHEOMlowervec(vec, site):
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

########################## Useful functions and variables #####################
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
def dagger(A):
    return transpose(conjugate(A))
def dissipator(L, rho):
    temp1 = matmul(L, matmul(rho, dagger(L)))
    temp2 = 1/2 * matmul(dagger(L), matmul(L, rho))
    temp3 = 1/2 * matmul(rho, matmul(dagger(L), L))
    return temp1 - temp2 - temp3
########## use superoperators on vectorized rhos to reduce runtime ###########
#### instead of reshaping vectorized and matrix rho back and forth
eye = np.identity(dim, dtype=complex)
eye_super = np.kron(eye, eye)
def leftmult_super(L):
    return np.kron(L, eye)
def rightmult_super(L):
    return np.kron(eye, transpose(L))
def commut_super(L):
    return leftmult_super(L) - rightmult_super(L)
def anticommut_super(L):
    return leftmult_super(L) + rightmult_super(L)
def dissipator_super(L):
    return matmul(leftmult_super(L), rightmult_super(dagger(L))) \
        - 1/2 * leftmult_super(matmul(dagger(L), L)) \
        - 1/2 * rightmult_super(matmul(dagger(L), L))
def dagger_super(vecrho):
    return np.reshape(dagger(np.reshape(vecrho, (dim,dim))), -1)
H_commut_super = commut_super(H)
total_diss_super = np.zeros((dim**2, dim**2), dtype=complex)
for L in L_emi:
    total_diss_super += dissipator_super(L)
L_abs_commut_super = commut_super(L_abs)
L_abs_dag_commut_super = commut_super(dagger(L_abs))
proj_commut_supers = [None]
proj_anticommut_supers = [None]
for i in range(1, sites+1):
    proj_commut_supers.append(commut_super(projs[i]))
    proj_anticommut_supers.append(anticommut_super(projs[i]))
# a vector v s.t. Tr(vec(x)) = v dot vec(x)
tr_super = np.zeros(rhodim, dtype=complex) 
for i in range(dim):
    tr_super[i*dim+i] = 1
############ Methods for accessing rhos in vectorized FS+HEOM state ###########   
def getrho(state, lvlvec):
    ''' Returns the auxiliary rho (vectorized) with index lvlvec 
    from vectorized HEOM state. '''
    assert len(lvlvec) == sites
    temp = HEOM_lvl_ind[tuple(lvlvec)]
    return state[temp*rhodim: (temp+1)*rhodim]
def addrho(state, lvlvec, rho):
    ''' Add rho (vectorized) to the auxiliary rho with HEOM index lvlvec. '''
    assert len(lvlvec) == sites
    temp2 = HEOM_lvl_ind[tuple(lvlvec)]
    state[temp2*rhodim: (temp2+1)*rhodim] += rho
    return None
def initHEOMvec(rhoinit):
    ''' Returns the vectorized HEOM state (with all the auxiliary rhos) '''
    rtn = np.zeros(HEOMlvls*rhodim, dtype=complex)
    temp = list(physvec)
    rhovec = np.reshape(rhoinit, -1)
    addrho(rtn, temp, rhovec)
    return rtn
############################## timederiv function #####################
def timederiv(t, state):
    '''The time derivative function during pulse. This acts on FS+HEOM vectors.
    Include both photon and phonon terms.'''
    rtn_state = np.zeros(HEOMlvls*rhodim, dtype=complex)
    for v in HEOM_lvl_vecs:
        temp = np.zeros(rhodim, dtype=complex)
        rhoin = getrho(state, v)
        temp += -1j * matmul(H_commut_super, rhoin)
        temp += matmul(total_diss_super, rhoin)
        sum_gamma = 0
        for s in range(1, sites+1):
            sum_gamma += HEOMsitelvl(v, s) * gammas[s]
        temp -= sum_gamma * rhoin
        if not cut_off_pulse:
            temp += timeprof(t).conjugate()*matmul(L_abs_commut_super, rhoin)
            temp -= timeprof(t)*matmul(L_abs_dag_commut_super, rhoin)
        else:
            if t < pulse_cutoff:
                temp += timeprof(t).conjugate()*matmul(L_abs_commut_super, rhoin)
                temp -= timeprof(t)*matmul(L_abs_dag_commut_super, rhoin)
        if special_terminators and sumHEOMsitelvl(v) == lvl: # terminators follow different EOMs
            for s in range(1, sites+1):
                lowervec = getHEOMlowervec(v,s)
                if lowervec != None:
                    rhoin = getrho(state, lowervec)
                    temp += HEOMsitelvl(v,s)*2*kT*\
                            matmul(proj_commut_supers[s], rhoin)
                    temp -= HEOMsitelvl(v,s)*1j*gammas[s]*\
                            matmul(proj_anticommut_supers[s], rhoin)
                rhoin = np.zeros(rhodim, dtype=complex)
                for t in range(1, sites+1):
                    if t == s:
                        rhoin2 = getrho(state, v)
                        rhoin += (2*kT*matmul(proj_commut_supers[t], rhoin2) \
                                  - 1j*gammas[t]*matmul(proj_anticommut_supers[t], rhoin2)) \
                                 *(HEOMsitelvl(v,t)+1)
                    else:
                        lowervec2 = getHEOMlowervec(v,t)
                        if lowervec2 != None:
                            rhoin2 = getrho(state, getHEOMhighervec(lowervec2, s))
                            rhoin += (2*kT*matmul(proj_commut_supers[t], rhoin2) \
                                      - 1j*gammas[t]*matmul(proj_anticommut_supers[t], rhoin2)) \
                                      * HEOMsitelvl(v,t)
                rhoin = rhoin / (sum_gamma + gammas[s])    
                temp -= lambdas[s]*matmul(proj_commut_supers[s], rhoin)
        else:
            for s in range(1, sites+1):
                lowervec = getHEOMlowervec(v,s)
                highervec = getHEOMhighervec(v,s)
                if lowervec != None:
                    rhoin = getrho(state, lowervec)
                    temp += HEOMsitelvl(v,s)*2*kT*\
                            matmul(proj_commut_supers[s], rhoin)
                    temp -= HEOMsitelvl(v,s)*1j*gammas[s]*\
                            matmul(proj_anticommut_supers[s], rhoin)
                if highervec != None:
                    rhoin = getrho(state, highervec)
                    temp -= lambdas[s]*matmul(proj_commut_supers[s], rhoin)
        addrho(rtn_state, v, temp)
    return rtn_state

##################################### flux ####################################
L_obs = []
for m in obs_modes:
    L_obs.append(collectiveL(matmul(dipoles, m)))
trL_supers = []
trLdL_supers = []
for L in L_obs:
    trL_supers.append(matmul(tr_super, leftmult_super(L)))
    trLdL_supers.append(matmul(tr_super, matmul(leftmult_super(dagger(L)),\
                                                leftmult_super(L))))
trL_abs_super = matmul(tr_super, leftmult_super(L_abs))
trLdL_abs_super = matmul(tr_super, matmul(leftmult_super(dagger(L_abs)),\
                                          leftmult_super(L_abs)))
def transflux(t, HEOMvec):
    ''' Returns the transmitted photon flux in units of ns^-1. 
    t is time in cm^-1. '''
    rtn = abs(timeprof(t))**2
    temp = timeprof(t).conjugate() * \
        matmul(trL_abs_super, getrho(HEOMvec, physvec))
    rtn += 2*temp.real
    rtn += matmul(trLdL_abs_super, getrho(HEOMvec, physvec))
    return rtn / cm_to_fs * 1e6
    
def scatflux(statevec):
    ''' Returns the scattered photon flux (ns^-1) as a list, whose elements
    are indexed the same way as obs_modes. '''
    rtn = np.zeros(len(obs_modes), dtype=complex)
    for i in range(len(obs_modes)):
        rtn[i] = matmul(trLdL_supers[i], getrho(statevec, physvec))
    return rtn / cm_to_fs * 1e6
################################## integrate ODE ##############################

def solvedynamics_full(flux=False):
    '''Solve dynamics while keeping all FS+HEOM rhos for all time points. '''
    starttime = time.time()
    initstate = initHEOMvec(rho_init)
    tpoints = np.arange(0, tf_cm, timestep_cm)   
    solution = solve_ivp(timederiv, (0,tf_cm), initstate, \
                         t_eval=tpoints, atol=abs_tol, rtol=rel_tol, max_step=maxstep, method=method)
    tpoints = solution.t
    rhos = solution.y[physical_ind*rhodim:(physical_ind+1)*rhodim,:]
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
#    pulsesections = int(pulse_cutoff//interval) + 1
    numsteps = int(interval // timestep_cm) + 2
    firstsection = True
    ti = 0
    initstate = initHEOMvec(rho_init)
    for s in range(section):
        print('progress:', str(s+1), '/', str(section))
        sec_tpoints = np.linspace(ti, ti+interval, num=numsteps)
        solution = solve_ivp(timederiv, (ti,ti+interval), initstate,\
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
        if s == section-1:
            finalstate = solution.y[:,-1]
        del solution
        gc.collect()
    print('solveivp time:', str(time.time()-starttime), 's')
    if flux:
        return tpoints*cm_to_fs, rhos, finalstate, transfluxes, scatfluxes
    return tpoints*cm_to_fs, rhos, finalstate

################################# Plotting ####################################
def plotrho(tpoints, rhopoints, row, col, form=None, part='R', yscale=1):
    if part!='R' and part!='I':
        raise Exception('part needs to be either "R" or "I".')
    if part == 'R':
        ypoints = np.real(rhopoints[dim*row+col, :])
    else:
        ypoints = np.imag(rhopoints[dim*row+col, :])
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
    pulsepts = timeprof(tpoints)
    plt.plot(tpoints * cm_to_fs, pulsepts*scale)
#    plt.fill_between((tpoints * cm_to_fs)[50:250], (pulsepts*scale)[50:250], color='lightgray')
########################## Display info and decay rate ########################
def displayparam():
    print('HEOM level =', str(lvl))
    print('coherent state amplitude =', str(alpha))
    print('sites =', str(sites))
    print('number of HEOM levels =', str(HEOMlvls))
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
    logfile.write('HEOM lvl: ' + str(lvl) + '\n')
    logfile.write('Temperature: ' + str(kT_K) + ' Kelvin \n')
    logfile.write('coherent state amplitude: ' + str(alpha) + '\n')
    logfile.write('Total number of HEOM levels: ' + str(HEOMlvls) + '\n')
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
    logfile.write('Modified HEOM terminator EOM: ' + str(special_terminators) + '\n')
    logfile.write('Index of refraction (water): ' + str(n_water) + '\n')
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


#plotrho(tpoints, rhopoints, 0, 2)
plotrho(tpoints, rhopoints, 2, 2)
plotrho(tpoints, rhopoints, 1, 2)
#plt.xlabel('time (fs)')
#plt.ylabel('site 1 prob')
#plt.title('10000000x Gamma')
#plt.legend(['Fock', 'coherent'])