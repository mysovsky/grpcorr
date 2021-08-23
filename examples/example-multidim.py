#!/usr/bin/python3

# example-multidim.py - example of finding & fixing point symmetry in multidimension

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

from sys import argv, path, exit
path += ['../']
from symmfinder import *
from polytope import symmetrize_points

if len(argv)==1:
    print('''Find point symmetry, correct symmetry group and symmetrize 
the points of arbitrary dimension red from file.
Usage: exmaple-multidim.py file epsilon algo
file - name of the file with the coordinates of points
epsilon (optional) - epsilon value for symmetry finder. Default = 1e-6
algo    (optional) - algorithm version. 0 is for fast algorithm, 1 is for 
                     slower version with "badness" tracking. Default = 0
''')
    exit(0)
    
# Read points
f = open(argv[1])
points = [ np.array([float(s) for s in l.split()]) for l in f]

epsilon = 1e-6
if len(argv) > 2:
    epsilon = float(argv[2])
    
algo = 0
if len(argv) > 3:
    algo = int(argv[3])

# Determine dimension and check if the points have it the same
dim = len(points[0])
for p in points:
    if len(p)!=dim:
        raise IndexError("Points have different dimension!")

for p in points:
    print((" {:10.5f}"*dim).format(*p))

P,G = symmetry_finder(points,dim,epsilon)
mtab = inclusive_closure(P,G)    
multab_group_correction(G,mtab,eps=1e-10)

points1 = symmetrize_points(points,P,G)

print("Symmetrized points:")
for p in points1:
    print((" {:10.5f}"*dim).format(*p))

print("Distances:")
for i in range(len(points1)):
    for j in range(i):
        print(i,j,la.norm(points1[i]-points1[j]))
    
