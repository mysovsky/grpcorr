#!/usr/bin/python3

# example-mol.py - example of group correction usage for randomly distorted symmetric molecules

# Copyright (C) 2019-2021 Andrey S. Mysovsky,
# Institute of Geochemistry SB RAS,

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

import sys
sys.path += ['../']

from grpcorr import *
from symmfinder import *

def rnd_displace(X,eps):
    for i in range(len(X['coord'])):
        for j in [0,1,2]:
            X['coord'][i][j] += eps*(2*random.random() - 1e0)

            
def rnd_axis():
    a = np.array([2*random.random()-1,2*random.random()-1,2*random.random()-1])
    return a/la.norm(a) 

def rnd_rotate(X,axis,phi):
    U = rot_mat(axis,phi)
    for i in range(len(X['coord'])):
        X['coord'][i] = np.dot(U,X['coord'][i])
        
def mol_copy(X):
    return {'atoms':X['atoms'].copy(), 'coord':[x.copy() for x in X['coord']]}

def lsf_corr_test(X,P,G,mtab):
    A = [X for p in P]
    B = [[X[p[i]] for i in range(len(X))] for p in P]
    #Q = make_Q(A,B)
    #for i in range(len(Q)):
    #    Q[i] = make_unitary(Q[i])
    abfit_group_correction(G,A,B,mtab, algo=0, maxit = 100, eps_mtab=1e-12, eps_diff=1e-12 )

def symmetrize_mol(mol,G,P):
    mol1 = [np.array([0e0,0e0,0e0]) for m in mol['coord']]
    for i in range(len(G)):
        for j in range(len(mol['coord'])):
            k = P[i][j]
            mol1[k] = mol1[k] + G[i].dot(mol['coord'][j])
    for i in range(len(mol1)):
        mol1[i] /= len(G)
    mol['coord'] = mol1
    return mol

def read_xyz(f):
    l = f.readline()
    n = int(l)
    l = f.readline()
    mol = {'atoms':[],'coord':[]}
    for i in range(n):
        s = f.readline().split()
        mol['atoms'].append(s[0])
        mol['coord'].append(np.array([float(s[i]) for i in [1,2,3]]))
    return mol

def print_xyz(mol):
    print(len(mol['atoms']))
    print()
    for i in range(len(mol['atoms'])):
        print('{:2s} {:10.5f} {:10.5f} {:10.5f}'.format(mol['atoms'][i],*mol['coord'][i]))

# Read molecule
fname = sys.argv[1]
f = open(fname)
mol = read_xyz(f)
f.close()

# Keep the copy of molecule
mol1 = mol_copy(mol)

# Apply random distortion if necessary
a = input('Apply random distortion?\n')
if 'y' in a or 'Y' in a:
    eps = float(input('Enter random distortion amplitute:\n'))
    rnd_displace(mol,eps)

print('Molecule:')
print_xyz(mol)

# Find approximate symmetry group
eps = float(input('Enter epsilon value for approximate symmetry search:\n'))

P,G = symmetry_finder(mol['coord'],3,eps)
mtab = inclusive_closure(P,G)

i = int(input(''' Approximate symmetry found. Next you can:
1. Apply multab based correction and symmetrize molecule;
2. Randomly rotate molecule and apply correction with rotation;
Your choice:
'''))

if i == 1:
    multab_group_correction(G,mtab,eps=1e-12)
elif i == 2:
    # Construct A and B vectors such that G[i]*A[i,j]=B[i,j]
    # We take atomic coordinates as A vectors
    # and the same coordinates reordered according to permutaion P[i] as B vectors
    A = [mol['coord'] for p in P]
    B = [[mol['coord'][p[i]] for i in range(len(mol['coord']))] for p in P]

    G = [ best_ab_transform(A[i],B[i]) for i in range(len(A))]

    # Randomly rotate the copy of mol
    phi  = pi*random.random()
    print('phi= ',180*phi/pi)
    rnd_rotate(mol,rnd_axis(),phi)

    # Call reconstruction with simultaneous fit
    abfit_group_correction(G,A,B,mtab)
    
mol = symmetrize_mol(mol,G,P)
print_xyz(mol)    
