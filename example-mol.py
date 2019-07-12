#!/usr/bin/python3

# example-mol.py - example of group correction usage for randomly distorted symmetric molecules

# Copyright (C) 2019 Andrey S. Mysovsky,
# (1) Institute of Geochemistry SB RAS,
# (2) Irkutsk National Research Technical University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from grpcorr import *
from misc import *
from math import sqrt, pi, sin, cos


def rnd_displace(X,eps):
    for i in range(len(X)):
        for j in [0,1,2]:
            X[i][j] += eps*(2*random.random() - 1e0)

            
def rnd_axis():
    a = np.array([2*random.random()-1,2*random.random()-1,2*random.random()-1])
    return a/sqrt(la.norm(a)) 


def rnd_rotate(X,axis,phi):
    U = rot_mat(axis,phi)
    for i in range(len(X)):
        X[i] = np.dot(U,X[i])

        
def mol_copy(X):
    return [np.array([y for y in x]) for x in X]

def lsf_corr_test(X,P,G,mtab):
    A = [X for p in P]
    B = [[X[p[i]] for i in range(len(X))] for p in P]
    #Q = make_Q(A,B)
    #for i in range(len(Q)):
    #    Q[i] = make_unitary(Q[i])
    abfit_group_correction(G,A,B,mtab, algo=0, maxit = 100, eps_mtab=1e-12, eps_diff=1e-12 )

    
def ch4():
    mol = [np.array([ 0.0,  0.0,  0.0]), # C\
           np.array([ 1.0,  1.0,  1.0]), # H\
           np.array([ 1.0, -1.0, -1.0]), # H\
           np.array([-1.0,  1.0, -1.0]), # H\
           np.array([-1.0, -1.0,  1.0])] # H
    a_ch4 = 1.087
    for i in range(len(mol)):
        mol[i] *= a_ch4/sqrt(3)        
    P = [permut([0,1,2,3,4])]
    symm_add(P, permut([0,2,3,4,1]))
    symm_add(P, permut([0,2,3,1,4]))
    return mol,P


def c2h6():
    a_c2h6 = 1.54
    b_c2h6 = 1.09
    theta_c2h6 = 109.5*pi/180
    r = b_c2h6*sin(theta_c2h6)
    z = b_c2h6*cos(theta_c2h6)
    mol = [np.array([  0.0,  0.0,  a_c2h6/2]), #C \
           np.array([  0.0,  0.0, -a_c2h6/2]), #C \
           np.array([  r/2,  r*sqrt(3)/2,  a_c2h6/2 - z]), #H \
           np.array([   -r,          0.0,  a_c2h6/2 - z]), #H \
           np.array([  r/2, -r*sqrt(3)/2,  a_c2h6/2 - z]), #H \
           np.array([ -r/2,  r*sqrt(3)/2, -a_c2h6/2 + z]), #H \
           np.array([    r,          0.0, -a_c2h6/2 + z]), #H \
           np.array([ -r/2, -r*sqrt(3)/2, -a_c2h6/2 + z])] #H     
    P = [permut([0,1,2,3,4,5,6,7])]
    symm_add(P,permut([1,0,5,7,6,3,2,4]))
    symm_add(P,permut([0,1,4,3,2,7,6,5]))
    return mol,P


def sf6():
    a = 1.564
    mol = [np.array([ 0.0, 0.0, 0.0]), #S \
           np.array([   a, 0.0, 0.0]), #F \
           np.array([  -a, 0.0, 0.0]), #F \
           np.array([ 0.0,   a, 0.0]), #F \
           np.array([ 0.0,  -a, 0.0]), #F \
           np.array([ 0.0, 0.0,   a]), #F \
           np.array([ 0.0, 0.0,  -a])] #F \
    P = [permut([0,1,2,3,4,5,6])]
    symm_add(P,permut([0,3,4,5,6,1,2]))
    symm_add(P,permut([0,1,2,5,6,4,3]))
    symm_add(P,permut([0,2,1,3,4,5,6]))
    return mol,P

def c20x():
    mol = [np.array([ 1.74385015, -0.71690039, -1.02287823]),
           np.array([ 1.95691872,  0.71148533, -0.51526413]),
           np.array([ 0.52826738, -0.71188623, -1.95330258]),
           np.array([-0.00993552,  0.71959842, -2.02072234]),
           np.array([ 0.87301957,  1.59929042, -1.1319657 ]),
           np.array([-0.55267286, -1.59452403, -1.32411677]),
           np.array([-1.75893357, -0.70853954, -1.00267832]),
           np.array([-1.42350344,  0.72166679, -1.43320424]),
           np.array([-0.0051479 , -2.14503835, -0.00483421]),
           np.array([-0.87301957, -1.59929042,  1.1319657 ]),
           np.array([-1.95691872, -0.71148533,  0.51526413]),
           np.array([ 1.41418137, -1.60263711,  0.18134145]),
           np.array([ 1.42350344, -0.72166679,  1.43320424]),
           np.array([ 0.00993552, -0.71959842,  2.02072234]),
           np.array([ 1.75893357,  0.70853954,  1.00267832]),
           np.array([ 0.55267286,  1.59452403,  1.32411677]),
           np.array([-0.52826738,  0.71188623,  1.95330258]),
           np.array([-1.74385015,  0.71690039,  1.02287823]),
           np.array([-1.41418137,  1.60263711, -0.18134145]),
           np.array([ 0.0051479 ,  2.14503835,  0.00483421])]
    P = [permut(list(range(20)))]
    symm_add(P,permut([1,4,0,2,3,11,8,5,12,13,9,14,15,16,19,18,17,10,6,7]))
    symm_add(P,permut([0,2,11,8,5,12,13,9,14,15,16,1,4,19,3,7,18,17,10,6]))
    symm_add(P,permut([0,11,2,5,8,3,7,6,4,19,18,1,14,15,12,13,16,17,10,9]))
    return mol, P

def c20():
    mol = [ np.array([ 1.888287176,        -0.784576197,        -0.877665063]),
            np.array([ 2.101476146,         0.644616649,        -0.369764126]),
            np.array([ 0.673400917,        -0.780625130,        -1.809100851]),
            np.array([ 0.136096344,         0.651564804,        -1.873867478]),
            np.array([ 1.018616025,         1.533654841,        -0.986661132]),
            np.array([-0.408933810,        -1.664401590,        -1.179103351]),
            np.array([-1.614226287,        -0.777654734,        -0.855052453]),
            np.array([-1.269400000,         0.649000000,        -1.276700000]),
            np.array([ 0.138310661,        -2.213513117,         0.140948497]),
            np.array([-0.730051407,        -1.667456812,         1.278390769]),
            np.array([-1.815065212,        -0.780177203,         0.662747364]),
            np.array([ 1.557441163,        -1.669951399,         0.326724938]),
            np.array([ 1.566280033,        -0.788591259,         1.578304447]),
            np.array([ 0.152965646,        -0.786768290,         2.166406658]),
            np.array([ 1.902249738,         0.641603235,         1.148196138]),
            np.array([ 0.696444693,         1.528149351,         1.470645798]),
            np.array([-0.385106344,         0.645012809,         2.100187134]),
            np.array([-1.601721766,         0.650051234,         1.171016325]),
            np.array([-1.269213001,         1.535272254,        -0.033093550]),
            np.array([ 0.149624744,         2.080106818,         0.151605329])]
    c = sum(mol)/len(mol)
    for i in range(len(mol)):
        mol[i] -= c
    P = [permut(list(range(20)))]
    symm_add(P,permut([1,4,0,2,3,11,8,5,12,13,9,14,15,16,19,18,17,10,6,7]))
    symm_add(P,permut([0,2,11,8,5,12,13,9,14,15,16,1,4,19,3,7,18,17,10,6]))
    symm_add(P,permut([0,11,2,5,8,3,7,6,4,19,18,1,14,15,12,13,16,17,10,9]))
    return mol, P

def print_group(G):
    for y in G:
        x=np.real(y)
        print('pq.matrix3f(pq.vector3f(',x[0,0],',',x[0,1],',',x[0,2],
              '),pq.vector3f(',x[1,0],',',x[1,1],',',x[1,2], 
              '),pq.vector3f(',x[2,0],',',x[2,1],',',x[2,2],')),\\')

def symm_mol(mol,G,P):
    mol1 = [np.array([0e0,0e0,0e0]) for m in mol]
    for i in range(len(G)):
        for j in range(len(mol)):
            k = P[i][j]
            mol1[k] = mol1[k] + np.matmul(np.asarray(G[i]),np.asarray(mol[j]))
    for i in range(len(mol1)):
        mol1[i] /= len(G)
    return mol1
            
from qpp import *


# Select one of the following c20(), sf6(), ch4(), c2h6()
# or construct your own molecule together with P - group of atomic permutations
# isomorphic to the molecule symmetry group
mol, P = c20()

# Keep the copy of molecule
mol1 = mol_copy(mol)

# Construct multiplication table
mtab = build_multab(P)

# Randomly deform the molecule
rnd_displace(mol,.5)

# Construct A and B vectors such that G[i]*A[i,j]=B[i,j]
# We take atomic coordinates as A vectors
# and the same coordinates reordered according to permutaion P[i] as B vectors
A = [mol for p in P]
B = [[mol[p[i]] for i in range(len(mol))] for p in P]

# Construct the sums of outer products of A and B
G = make_Q(B,A)

# Unitarize them
for i in range(len(G)):
    G[i] = make_unitary(G[i])

# Scenario 1. Comment this call if you need scenario 2
# Initial approximation is ready. Now call the reconstruction
#multab_group_correction(G,mtab,eps=1e-12)

# Scenario 2
# Randomly rotate the copy of mol
phi  = pi*random.random()
print('phi= ',180*phi/pi)
rnd_rotate(mol1,rnd_axis(),phi)

# Randomly deform it
rnd_displace(mol1, .5)

# Construct A and B vectors again for mol1
A = [mol1 for p in P]
B = [[mol1[p[i]] for i in range(len(mol1))] for p in P]

# Call reconstruction with simultaneous fit
abfit_group_correction(G,A,B,mtab)
