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


import sys
sys.path += ['../']
from polytope import *


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

