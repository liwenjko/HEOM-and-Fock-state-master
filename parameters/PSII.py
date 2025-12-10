# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:44:25 2019

This contains an object oriented representation of the PSII supercomplex

@author: rlcook
"""
import io
import numpy as np
from numpy import sin, cos
from scipy.io import loadmat

import csv

# define some useful constants
invcm = 1 # fundamental unit
#a0 = 5.29177211e-9/invcm      # google quotes 1 bohr radius as 5.29... cm
#alpha = 1/137.036          # fine structure constant
#Debye = 1/2.541746           # 1 debye /(e*a0) i.e. 1 debye in atomic units (e*a0)

CoulombEnergy =  5034.11692*invcm #  e^2/ angstrom /(4 pi epsilon0) in units of 1/cm


class PSII(object):
    def __init__(self):
        
        self.parse_site_labels()
        self.read_site_names()
        
        self.n_chl = len(self.chl_names)
        
        self.ind_dict = dict()
        self.ind2names = []

        for i in range(self.n_chl):
            label = self.cpx_labels[i]
            name = self.chl_names[i]
            self.ind_dict.setdefault(label, dict())[name] = i
            self.ind2names.append((label, name))
        
        self.ind2names = np.array(self.ind2names)
        self.masks = dict()
        
        for label in self.ind_dict.keys():
            self.masks[label] = np.array([ l == label for l in self.cpx_labels] )
        
        
        #make masks for the LHCII trimers
        tri_labels = [['LHCIIA_Top','LHCIIB_Top','LHCIIC_Top' ],
                      ['LHCIIA_Bot','LHCIIB_Bot','LHCIIC_Bot' ],
                      ['LHCIIa_Top','LHCIIb_Top','LHCIIc_Top' ],
                      ['LHCIIa_Bot','LHCIIb_Bot','LHCIIc_Bot' ]]
        
        tri_keys = ['LHCIIABC_Top', 'LHCIIABC_Bot', 'LHCIIabc_Top', 'LHCIIabc_Bot']
        for i, key in enumerate(tri_keys):
            self.masks[key] = np.sum(np.block([ [self.masks[l]] for l in tri_labels[i] ] ), axis = 0, dtype = bool)
        
        self.masks['PSII'] = np.full((self.n_chl,), True)
        
        self.scrape_PSII_mat() # load the (CP29 deleted) Hamiltonian
        self.read_dipoles()
        self.read_positions()

        #make masks for the 'stromal' and 'lumenal' sites
        self.masks['lumenal'] = self.positions[:, -1] >= 0.0
        self.masks['stromal'] = self.positions[:, -1] < 0
        
        
        #make masks for the 'top' and 'bottom' halves
        top_mask = np.full_like(self.masks['RC_Top'], False)
        bot_mask = np.full_like(self.masks['RC_Top'], False)
        for name in self.ind_dict.keys():
            if name[-3:] == 'Top':
                top_mask += self.masks[name]
            if name[-3:] == 'Bot':
                bot_mask += self.masks[name]
        
        self.masks['Top'] = top_mask
        self.masks['Bot'] = bot_mask
        
        #store the order of the common names of the reaction center names
        self.RC_Top_names = ['P_D1', 'Chl_D1', 'ChlZ_D1', 'Chl_D2', 'P_D2', 'ChlZ_D2', 'Pho_D1', 'Pho_D2']
        self.RC_Bot_names = ['P_D1', 'Chl_D2', 'Chl_D1', 'ChlZ_D1', 'P_D2', 'ChlZ_D2', 'Pho_D1', 'Pho_D2']
        
        #make masks that isolate only the D1 and D2 sites for top and bottom halves
        D1D2_Energies = np.array([ E ==  14992 or E == 14750 for E in np.diag(self.H) ])
        
        D1D2_Top = np.all( np.array( [ [v1,v2] for v1,v2 in zip(D1D2_Energies, self.masks['RC_Top']) ] ), axis = 1 )
        D1D2_Bot = np.all( np.array( [ [v1,v2] for v1,v2 in zip(D1D2_Energies, self.masks['RC_Bot']) ] ), axis = 1 )
        self.masks['D1D2_Top'] = D1D2_Top
        self.masks['D1D2_Bot'] = D1D2_Bot
        
        
    def gen_cpx_mask(self, *cpx_names):
        
        cpx_array = np.stack( [ self.masks[key] for key in cpx_names] )

        cpx_mask = np.any(cpx_array, axis = 0)
        
        return cpx_mask

    
    def rotate_ZYZ(self, euler_angles):
        def Rz(th):
            return np.array([[cos(th), -sin(th), 0.],
                              [sin(th), cos(th), 0.],
                              [0.,0.,1.]])
            
        def Ry(th):
            return np.array([[cos(th), 0, sin(th), ],
                              [0., 1., 0.],
                              [-sin(th), 0., cos(th)]])
            
        R = Rz(euler_angles[2]) @ Ry(euler_angles[1]) @ Rz(euler_angles[0])
        
        self.positions = self.positions @ R.T
        
        self.dipoles = self.dipoles @ R.T
        
    def translate(self, X0):
        self.positions = self.positions + np.tile( X0, [self.n_chl, 1])
        
    def orient_to_XYZ(self):
        self.rotate_ZYZ([0, np.pi, 0])

        ctr = np.mean(self.positions, axis = 0)
        zList = self.positions[:, -1]
    
        ctr[-1] = np.mean( (min(zList[ zList > 0] ), max(zList[zList < 0]) ) )
    
        self.translate(-ctr)
    
        self.rotate_ZYZ([-33*np.pi/180, 0, 0])


    def parse_site_labels(self):
        # copy and paste the raw text from SiteLabels.csv
        SiteLabels_raw = "R,C,_,T,o,p,R,C,_,T,o,p,R,C,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,7,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,C,P,4,3,_,T,o,p,R,C,_,T,o,p,R,C,_,T,o,p,R,C,_,T,o,p,R,C,_,B,o,t,R,C,_,B,o,t,R,C,_,B,o,t,R,C,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,7,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,C,P,4,3,_,B,o,t,R,C,_,B,o,t,R,C,_,B,o,t,R,C,_,T,o,p,R,C,_,T,o,p,R,C,_,B,o,t,R,C,_,B,o,t,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,C,P,2,9,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,A,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,B,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,C,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,a,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,b,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,L,H,C,I,I,_,c,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,6,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,4,_,T,o,p,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,C,P,2,9,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,A,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,B,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,C,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,a,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,b,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,L,H,C,I,I,_,c,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,6,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t,C,P,2,4,_,B,o,t"
        
        SiteLabels_raw = SiteLabels_raw.replace(',', '')
        SiteLabels_raw = SiteLabels_raw.replace('LHCII_', 'LHCII')
        SiteLabels = []
        
        
        remainder = SiteLabels_raw
        sep = '_'
        while len(sep) > 0:
            part, sep, remainder = remainder.partition('_')
            if remainder[:3] == 'Top' or remainder[:3] == 'Bot':
                SiteLabels.append(part+'_'+remainder[:3])
                remainder = remainder[3:]
        
        self._site_labels_full = np.array(SiteLabels)

        #swap the top and bottom labels on LHCIIa, LHCIIb, LHCIIc, CP29, CP24
        top_labels = np.array(['LHCIIa_Top','LHCIIb_Top','LHCIIc_Top', 'CP29_Top', 'CP24_Top'])
        bot_labels = np.array(['LHCIIa_Bot','LHCIIb_Bot','LHCIIc_Bot', 'CP29_Bot', 'CP24_Bot'])
        for i, label in enumerate(self._site_labels_full):
            if np.any(top_labels == label):
                self._site_labels_full[i] = label[:-3] + 'Bot'
            elif np.any(bot_labels == label):
                self._site_labels_full[i] = label[:-3] + 'Top'
            

        
        self._site_mask = np.full( (len(self._site_labels_full),), True)
        deletes = [78, 204] # delete CHL B 605 from CP 29 Top and Bottom.  CP 29 is a copy of LHCII, sans CHL 605
        for d in deletes:
            self._site_mask[d] = False
            
            
        self.cpx_labels = self._site_labels_full[self._site_mask]
        
    def read_positions(self):
        x = []
        y = []
        z = []
        with open('PSIIcoords_cNov.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                x.append(np.float_(row['x']))
                y.append(np.float_(row['y']))
                z.append(np.float_(row['z']))
        
        self.positions = np.array([(x,y,z) for x,y,z in zip(x,y,z)])*1e-8 # convert to cm
        
                
    def read_site_names(self):
        file = io.open('PSIILabels_cNov.csv')
        
        filestr = file.read()
        file.close()
        
        filestr = filestr.replace(' ', '')
        filestr = filestr.replace(',', ' ')
        
        names = []
        sep = '\n'
        remainder = filestr
        while len(sep) > 0:
            part, sep, remainder = remainder.partition('\n')
            if len(part) > 0:
                names.append(part)        
        
        self.chl_names = names

    def scrape_PSII_mat(self):
        self._PSII_mat_raw = loadmat('PlotCouplingOnPSII.mat')    
#        self.positions = self._PSII_mat_raw['MgPos']
        self.H = self._PSII_mat_raw['Hamiltonian']
#        self.dipole = np.zeros_like(self._PSII_mat_raw['Qy'])
#        for i in range(len(self.dipole)):
#            Qy = self._PSII_mat_raw['Qy'][i, :]
#            self.dipole[i, :] = Qy/np.sqrt( Qy @ Qy )

    def calculate_dipole_dipole(self):
        Jij = np.zeros((self.n_chl, self.n_chl))
        
        for i in range(self.n_chl):
            for j in range(i+1, self.n_chl):
                R = (self.positions[i, :] - self.positions[j,:])*1e8
                d = np.sqrt(R @ R) #convert to anstroms
                Jij[i,j] = CoulombEnergy*( (self.dipoles[i,:] @ self.dipoles[j,:] )/d**3 - 3*(self.dipoles[i,:] @ R) * (self.dipoles[j,:] @ R)/d**5)
                Jij[j,i] = Jij[i,j]
                
        self.Jij = Jij


            
    def read_dipoles(self, renormalize = True):
        Dx = []
        Dy = []
        Dz = []
        with open('QyDipoles_cNov.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                Dx.append(np.float_(row['Dx']))
                Dy.append(np.float_(row['Dy']))
                Dz.append(np.float_(row['Dz']))
        
        self.dipoles = np.array([(dx,dy,dz) for dx,dy,dz in zip(Dx,Dy,Dz)]) 
        
        if renormalize:
            norms = np.sqrt(np.diag(self.dipoles @ self.dipoles.T))
            factors = np.tile(np.round(norms, decimals = 3)/norms, [3,1]).T
            self.dipoles = self.dipoles*factors
            
            
        
if __name__ == '__main__':
    psii = PSII()
    
            
            
        