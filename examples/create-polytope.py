#!/usr/bin/python3

# create-polytope.py - simple polytopes creation utility

# Copyright (C) 2021 Andrey S. Mysovsky,
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

import random
from math import sqrt
from sys import exit
import numpy as np
import numpy.linalg as la
import sys
sys.path += ['../']
from misc import permut, symm_add

def random_distortion(points,eps):
    for i in range(len(points)):
        for j in range(len(points[i])):
            points[i][j] += 2*eps*(random.random() - 5e-1)

def plusminus(points, i, dim):
    if i==dim:
        return points
    nextp = plusminus(points, i+1, dim)
    newp = []
    for j in range(len(nextp)):
        newp.append(nextp[j].copy())
        nextp[j][i] *= -1
        newp.append(nextp[j].copy())
    return newp
            
def hypercube(a,dim):
    p0 = np.array([a/2 for i in range(dim)])
    return plusminus([p0],0,dim)

def hypercube_dual(a,dim):
    return [np.array([ s*a if j==i else 0e0 for j in range(dim)]) for i in range(dim) for s in [-1.,1.]]

def simplex(a,dim):
    if dim==1:
        return [np.array([a]), np.array([-a])]
    #else
    s1 = simplex(1,dim-1)
    s = [np.array([*p, -1./sqrt(dim**2 - 1)]) for p in s1] + \
        [np.array([0e0]*(dim-1) + [dim/sqrt(dim**2 - 1)])]
    r = la.norm(s[0])
    return [a*p/r for p in s]
    

def dodecahedron(a):
    sg = [-1.,1.]
    points = [np.array([s1,s2,s3]) for s1 in sg for s2 in sg for s3 in sg]
    h = (sqrt(5)-1)/2
    points += [np.array([0, s1*(1+h), s2*(1-h**2)])  for s1 in sg for s2 in sg]
    points += [np.array([s1*(1+h), s2*(1-h**2), 0])  for s1 in sg for s2 in sg]
    points += [np.array([s2*(1-h**2), 0, s1*(1+h)])  for s1 in sg for s2 in sg]
    for i in range(len(points)):
        points[i] *= a/sqrt(3)
    return points

def icositetrachoron(a):
    sg = [-1.,1.]
    points = [ np.array([s*a if j==i else 0 for j in range(4)]) for i in range(4) for s in sg] +\
    [ .5*a*np.array([s1,s2,s3,s4]) for s1 in sg for s2 in sg for s3 in sg for s4 in sg]
    return points

def hexacosichoron(a):
    points = icositetrachoron(a)
    P = [permut([0,1,2,3])]
    symm_add(P,permut([1,2,0,3]))
    symm_add(P,permut([0,3,1,2]))
    sg = [-1.,1.]
    v = [.5, (sqrt(5)+1)/4, (sqrt(5)-1)/4]
    vv = [[v[0]*s0, v[1]*s1, v[2]*s2, 0e0] for s0 in sg for s1 in sg for s2 in sg]
    points += [ a*np.array([x[p[0]],x[p[1]],x[p[2]],x[p[3]]]) for x in vv for p in P]
    return points

i = int(input(
'''Menu:
1. Hypercube
2. Hypercube dual polytope
3. Simplex
4. Dodecahedron (3D)
5. Icositetrachoron (4D)
6. Hexacosichoron (4D)
'''))

if i in [1,2,3]:
    dim = int(input("Enter dimension: "))

a = float(input("Enter polytope radius (i.e. distance between vertices and coordinate centre): "))

if i==1:
    points = hypercube(2*a/sqrt(dim),dim)
elif i==2:
    points = hypercube_dual(a,dim)
elif i==3:
    points = simplex(a,dim)
    pass
elif i==4:
    points = dodecahedron(a)
    dim = 3
elif i==5:
    points = icositetrachoron(a)
    dim = 4
elif i==6:
    points = hexacosichoron(a)
    dim = 4
else:
    print("Unrecognized choice")
    exit(1)

eps = float(input("Enter the random distortion amplitude: "))
        
random_distortion(points,eps)

fname = input("Enter filename to save: ")
f = open(fname,'w')
for p in points:
    print((" {:10.5f}"*dim).format(*p), file = f)
f.close()

