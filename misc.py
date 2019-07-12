# SYMM - auxillary stuff to work with symmetry

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

from math import cos,sin,sqrt
import numpy as np

class permut:
    def __init__(self,li):
        if type(li)!=list:
            raise TypeError('permut constructor: needs a list of integers')
        self.p = li

    def __getitem__(self,i):
        return self.p[i]

    def __setitem__(self,i,j):
        self.p[i] = j

    def __mul__(self,perm2):
        if len(self.p)!=len(perm2.p):
            raise IndexError('Permutations have different dimensions')
        res = []
        for i in range(len(self.p)):
            #res.append(perm2[self.p[i]])
            res.append(self[perm2[i]])
        return permut(res)

    def __len__(self):
        return len(self.p)

    def __eq__(self,perm2):
        if len(self) != len(perm2):
            return False
        for i in range(len(self)):
            if self[i] != perm2[i]:
                return False
        return True

    def __ne__(self,perm2):
        return not self==perm2

    def __str__(self):
        return 'permut(' + str(self.p) +')'

    def __repr__(self):
        return 'permut(' + str(self.p) +')'
    
#---------------------------------------------------------

def overloader(funlist,args,kwds,msg="Invalid arguments"):
    for f in funlist:
        try:
            res = f(*args,**kwds)
            return res
        except TypeError:
            pass
    raise TypeError(msg)
 
#---------------------------------------------------------

def symm_add(group,op):
    if op in group:
        return
    newelems = [op]
    while newelems != []:
        newnewelems = []
        #print "new ", newelems
        for g1 in newelems:
            for g2 in group+newelems:
                h = g1*g2
                if (not h in group) and (not h in newelems) and (not h in newnewelems):
                    newnewelems.append(h)
                h = g2*g1
                if (not h in group) and (not h in newelems) and (not h in newnewelems):
                    newnewelems.append(h)
        #print group, newelems, newnewelems
        group.extend(newelems)
        newelems = newnewelems
        
#---------------------------------------------------------

def build_multab(group):
    N = len(group)
    tab = []
    for i in range(N): tab.append(list([0]*N))
    #print tab
    for i in range(N):
        for j in range(N):
            #print i,j,group.index(group[i]*group[j])
            tab[i][j] = group.index(group[i]*group[j])
    return tab

#---------------------------------------------------------

def symm_order_by_multab(multab,i):
    #print "by multab call"
    if (type(multab) is list) and (type(multab[0]) is list) and (type(multab[0][0]) is int):
        n=0;
        j=i
        while j!=0:
            j = multab[j][i]
            n = n + 1
        return n + 1
    else:
        #print "by multab TE"
        raise TypeError()

def symm_order_by_element(A):
    #print "by element"
    n=0
    B=A*A
    while B!=A:
        B = B*A
        n = n + 1
    return n + 1

def symm_order_by_index(group,i):
    return symm_order_by_element(group[i])

def symm_order(*args,**kwds):
    return overloader([symm_order_by_multab,symm_order_by_element,symm_order_by_index],args,kwds, \
                      "qpp: invalid arguments in symm_order call"+str(args)+str(kwds))

#---------------------------------------------------------

def rot_mat(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

