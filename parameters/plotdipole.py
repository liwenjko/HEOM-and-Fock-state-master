#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:07:12 2020

@author: LiwenKo
"""

from PSII import PSII
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

psii = PSII()
m = psii.masks['LHCIIA_Top']
H_sys = psii.H[m,:][:,m]
dipoles = -psii.dipoles[m,:]
positions = psii.positions[m,:]
chl_names = np.array(psii.chl_names)[m]
site_energies = np.diag(H_sys)

R = np.array([[0.7415613, -0.51397872, 0.43117596],
              [-0.22095305, 0.41973122, 0.88034394],
              [ 0.63345607, 0.74809864, -0.19769127]])
#R = np.array([[0.7415613, -0.22095305, 0.63345607],
#              [-0.51397872, 0.41973122, 0.74809864],
#              [ 0.43117596, 0.88034394, -0.19769127]])
#test = np.array([-0.22095305, 0.41973122, 0.88034394])
dipoles = np.matmul(dipoles, np.transpose(R))
positions = np.matmul(positions, np.transpose(R))

pos_x = positions[:,0] * 1e8 # in Angstrom
pos_y = positions[:,1] * 1e8
pos_z = positions[:,2] * 1e8

pos_x = pos_x - np.average(pos_x)
pos_y = pos_y - np.average(pos_y)
pos_z = pos_z - np.average(pos_z)

dip_x = dipoles[:,0] # in Debye
dip_y = dipoles[:,1]
dip_z = dipoles[:,2]
#
#
#

stroma = np.array([1,2,3,8,9,10,11,12])
lumen = np.array([4,5,6,7,13,14])
stroma_mask = np.zeros(len(site_energies), dtype=bool)
for s in stroma:
    stroma_mask[s-1] = True
lumen_mask = np.zeros(len(site_energies), dtype=bool)
for l in lumen:
    lumen_mask[l-1] = True

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.quiver(pos_x[stroma_mask], pos_y[stroma_mask], pos_z[stroma_mask]\
          , dip_x[stroma_mask], dip_y[stroma_mask], dip_z[stroma_mask]\
          , arrow_length_ratio=0.5, color='tab:blue')
ax.quiver(pos_x[lumen_mask], pos_y[lumen_mask], pos_z[lumen_mask]\
          , dip_x[lumen_mask], dip_y[lumen_mask], dip_z[lumen_mask]\
          , arrow_length_ratio=0.5, color='tab:red')
ax.set_xlabel('x (A)')
ax.set_ylabel('y (A)')
ax.set_zlabel('z (A)')
ax.set_xlim(-25,25)
ax.set_ylim(-25,25)
ax.set_zlim(-25,25)
#ax.set_title('LHCII monomer dipole orientations')
ax.grid(True)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.view_init(17,270+44)
ax.legend(['stromal', 'lumenal'])

for i in range(14):
    ax.text(pos_x[i], pos_y[i], pos_z[i]+0.5, str(i+1))
    
    
#fig, ax = plt.subplots()
#temp = np.square(dip_z)
#alphas = temp/np.max(temp)
#ax.set_xlim([-12, 12])
#ax.set_ylim([-15,15])
#for i in range(1, 15):
#    if i in stroma:
#        ax.quiver(pos_x[i-1], pos_y[i-1], dip_x[i-1], dip_y[i-1], color='tab:blue', alpha=alphas[i-1])
#    elif i in lumen:
#        ax.quiver(pos_x[i-1], pos_y[i-1], dip_x[i-1], dip_y[i-1], color='tab:red', alpha=alphas[i-1])
#for i in range(14):
#    if i+1==1 or i+1==2:
#        ax.text(pos_x[i], pos_y[i]+0.3, str(i+1))
#    elif i+1==3 or i+1==10 or i+1==11 or i+1==12:
#        ax.text(pos_x[i]+0.5, pos_y[i], str(i+1))
#    elif i+1==5:
#        ax.text(pos_x[i]-0.5, pos_y[i], str(i+1))
#    else:
#        ax.text(pos_x[i], pos_y[i], str(i+1))
#ax.set_xlabel('x (A)')
#ax.set_ylabel('y (A)')
    
#def DL_SpectralDensity(omega):
#    return 2*37*30*omega/(omega**2+30**2)
#omegapoints = np.arange(0,200,0.1)
#Jpoints = [DL_SpectralDensity(w) for w in omegapoints]
#plt.plot(omegapoints, Jpoints)
#plt.title('Drude-Lorentz Spectral Density')
#plt.xlabel('$\omega$ ($cm^{-1}$)')
#plt.ylabel('$J(\omega)$ (1/$cm^{-1}$)')

#plt.vlines(site_energies, ymin=0, ymax=1)
#from math import e
#def pulseshape(omega):
#    return e**(-(omega-np.average(site_energies))**2/(4*np.var(site_energies)))
#omegapoints = np.arange(13000, 18000, 1)
#pulsepoints = [pulseshape(w) for w in omegapoints]
#plt.plot(omegapoints, pulsepoints)
#plt.xlim([14000, 17000])
#plt.xlabel('frequency ($cm^{-1}$)')


#shifted_text = np.array([9, 12, 13])
#fig, ax = plt.subplots()
#for i in range(14):
#    xpoints = np.array([0,0.6])
#    ypoints = np.array([site_energies[i], site_energies[i]])
#    ax.plot(xpoints, ypoints, color='k')
#    if (i+1) in shifted_text:
#        ax.text(-0.13, site_energies[i]-10, str(i+1))
#    else:
#        ax.text(-0.2, site_energies[i]-10, str(i+1))
#ax.set_xlim([-0.5,2])
#ax.set_ylim([15000, 16000])
#ax.set_ylabel('site energy (cm$^{-1}$)')
#
#from math import e
#def freq_dist(omega):
#    return e**(-2*(omega-15445.07)**2/299.1158**2)
#ypoints = np.arange(15000, 16000, 1)
#xpoints = [0.8+0.5*freq_dist(omega) for omega in ypoints]
#ax.plot(xpoints, ypoints)
#fig.set_figheight(5.5)
#ax.tick_params(axis='x', which='both', labelbottom=False)

#import pickle as pkl
#pkl.dump(fig, open('LHCII_monomer_site_energies_06042021.pkl', 'wb'))