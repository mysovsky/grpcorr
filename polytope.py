# polytope.py - auxillarly stuff

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
from math import sqrt, acos, pi
from sys import exit
import numpy as np
import numpy.linalg as la
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

def symmetrize_points(points,P,G):
    dim = len(points[0])
    points1 = [np.array([0e0 for i in range(dim)]) for p in points]           
    for i in range(len(G)):
        for j in range(len(points)):
            k = P[i][j]
            points1[k]  += G[i].dot(points[j])/len(G)
    return points1

def analyze_matrix3(G, eps = 1e-10):
    inversion = np.linalg.det(G) < 0e0
    G1 = G.copy()
    if inversion: G1 *= -1
    A = 0.5*(G1 - G1.transpose())
    trace = G1[0,0] + G1[1,1] + G1[2,2]
    cos = 0.5*(trace - 1e0)
    if cos > 1e0: cos = 1e0
    if cos < -1e0: cos = -1e0
    phi = acos(cos)
    axis = np.array([-A[1,2], A[0,2], -A[0,1] ])
    nax = np.linalg.norm(axis)
    if nax > eps:
        axis /= nax
    elif phi < eps:
        axis = np.array([0., 0., 1.])
    else:
        G1 += np.identity(3)
        axis = max([G1[:,i] for i in [0,1,2]], key = lambda v : np.linalg.norm(v))
        axis /= np.linalg.norm(axis)
    return {'inversion': inversion, 'angle':phi, 'order': 2*pi/phi if phi > eps else 1, 'axis': axis}
