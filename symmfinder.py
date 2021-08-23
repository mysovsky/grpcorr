# SYMMFINDER - module implementing symmetry finder in arbitrary dimension

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

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from grpcorr import *
from misc import *
from math import acos, pi
import time

def pkey(a,b):
    return (a,b) if a>b else (b,a)

def arccos(x):
    if x > 1e0:
        return 0e0
    elif x < -1e0:
        return pi
    else:
        return acos(x)

def gramm_schmidt_projector(projector,point):
    tp = point - projector.dot(point)
    tp /= la.norm(tp)
    return projector + np.outer(tp,tp)

def lindepcy(projector,point,eps=1e-6):
    pp = projector.dot(point)
    return la.norm(pp-point) < eps

def match_possible(i,idx,idxprime,pt2pt,pair2pair):
    possible = []
    for ip in pt2pt[i]:
        go = True
        for k in range(len(idx)):
            j  = idx[k]
            jp = idxprime[k]
            if not (ip,jp) in pair2pair[i,j]:
                go = False
                break
        if go:
            possible.append(ip)
    return possible

def match_rest(idx_subst, possible_subst, badness_list, points,  algo_fast, epsilon):
    # Match all other points that have only one matcing possibility
    for i in range(len(points)):
        if not i in idx_subst and len(possible_subst[i])==1:
            ip = possible_subst[i][0]
            if ip in [idx_subst[k] for k in idx_subst]:
                return [],[]
            idx_subst[i] = ip
    # This points are not yet matched
    idx_rest = [i for i in range(len(points)) if not i in idx_subst]
    idx_already = [idx_subst[i] for i in idx_subst]
    # Find unitary transformation
    a = [points[i]            for i in idx_subst]
    b = [points[idx_subst[i]] for i in idx_subst]
    U = best_ab_transform(a,b)
    # Try to match the rest
    fail = False
    #print(idx_subst, idx_rest, idx_already)
    for i in idx_rest:
        p = U.dot(points[i])
        found = False
        for ip in possible_subst[i]:
            if not ip in idx_already:
                #print(i,ip,la.norm( points[ip] - p ))
                if la.norm( points[ip] - p ) < epsilon:
                    found = True
                    break
        if found:
            idx_subst[i] = ip
            idx_already.append(ip)
        else:
            #print('\nfailed ab transform ', idx_subst, idx_rest, idx_already)
            #print(epsilon)
            #print(i,possible_subst[i])
            #print(a)
            #print(b)
            #print(U)
            #print(p,points[i])
            #print(la.norm( points[i] - p ))
            fail = True
            if not algo_fast:
                try:
                    badness_list[i] += 1
                except KeyError:
                    badness_list[i] = 1
                print(badness)
            else:
                break
    if fail: return [],[]
    return [tuple(idx_subst[i] for i in range(len(points)))],[U]
    
def match_next(idx_subst, possible_subst, badness_list, projector, points, dim, pair2pair,
               algo_fast, epsilon, nfound=0):
    if len(idx_subst) == dim:
    # Finalize matching
        perm, op = match_rest(idx_subst, possible_subst, badness_list, points, algo_fast, epsilon)
        return perm, op
    # Check first for linear dependence
    indep_points = [i for i in possible_subst if not lindepcy(projector,points[i],epsilon)]
    # No suitable points left? Then actual dimesion of this points is lower than dim
    if len(indep_points) == 0:
        raise la.LinAlgError("Actual dimesion of your set of points is lower than dim. Try using dim=" +
                             str(len(idx_subst)) +
                             ". Note in this case you will get only symmetry subgroup operating inside the linear  subspace spanned by your points. The full symmetry group must include also all symmetry operations leaving that subspace invariant and might be infinite")
    minmax = min if algo_fast else max
    i = minmax(indep_points, key = lambda k : len(possible_subst[k]))
    projector_new = gramm_schmidt_projector(projector,points[i])
    res_perms = []
    res_symmops = []
    for ip in possible_subst[i]:
        idx_subst_new = idx_subst.copy()
        idx_subst_new[i] = ip
        possible_subst_new = {}
        nomatch = False
        for j in possible_subst:
            if j != i:
                possible_subst_new[j] = [jp for jp in possible_subst[j] if (ip,jp) in pair2pair[i,j] ]
                if algo_fast and possible_subst_new[j] == []:
                    nomatch = True
                    break
        if not algo_fast:
            for j in possible_subst_new:
                if possible_subst_new[j] == []:
                    if j in badness_list:
                        badness_list[j] += 1
                    else:
                        badness_list[j] = 1
                    nomatch = True
        if nomatch:
            continue
        #print('current: ',idx_subst_new)
        #print('possibilities: ',possible_subst_new)
        perms, ops =  match_next(idx_subst_new, possible_subst_new, badness_list, projector_new, points, dim,
                                 pair2pair, algo_fast, epsilon, nfound + len(res_perms))
        res_perms   += perms
        res_symmops += ops
        
        print('\r', end='')
        print("{:10d} approximate symmetry operations found".format(nfound+len(res_perms)), end="", flush = True)
    return res_perms, res_symmops
        
def symmetry_finder(points,dim,epsilon, algo = 0, badness_list = None):
    '''
    Driver routine that tries to find point symmetry of given list of vectors
    in arbitrary dimension space
    points (IN, list of vectors)    - the list of points
    dim (IN, int)                   - the dimension of the space
    epsilon(IN, REAL)               - the tolerance for symmetry operations. That is, if symmetry operation
                                      G must turn the vectors points[i] into points[j], then inequality
                                      | G*points[i] - points[j] | < epsilon must be fulfilled
    algo(IN, int)                   - Algorithm selection
    badness_list(OUT, dict int:int) - "Badness" ascribed to each point. Removing the points with highest badness
                                      might inscrease symmetry
    algo = 0 - fast algorithm 
    algo = 1 - slow algorithm with badness tracking
               Badness tracking ascribes an integer value to each point (e.g. bandess_list[i] for
               point number i) which is the number of possible symmetry operations rejected 
               because of this point. Deleting the points with highest "badness" might make the 
               points distribution more symmetric
    '''
    N = len(points)
    algo_slow      = algo == 1
    algo_fast      = algo == 0

    if algo_slow and badness_list==None:
        raise ValueError("For slow algorithm please provide badness_list as empty dictionary for badness tracking")
    
    t0 = time.process_time()

    # Construct dot products matrix
    S = np.asmatrix(np.zeros((N,N)))
    for i in range(N):
        for j in range(i+1):
            S[i,j] = S[j,i] = points[i].dot(points[j])
    # Construct absolute values array
    R = np.zeros((N))
    for i in range(N):
        R[i] = sqrt(S[i,i])

    t1 = time.process_time()
    print('Dot products matrix: {:8.3f} sec CPU time'.format(t1-t0))

    # For each point construct the list of points in which it can be transformed
    pt2pt = [[] for i in range(N)]
    for i in range(N):
        pt2pt[i].append(i)
        for j in range(i):
            if abs(R[i]-R[j]) < epsilon:
                pt2pt[i].append(j)
                pt2pt[j].append(i)
                
    t2 = time.process_time()
    print('Point to point list: {:8.3f} sec CPU time'.format(t2-t1))

    # Angle matrix
    alpha = { (i,j):arccos(S[i,j]/(R[i]*R[j])) for i in range(N) for j in range(i) }
    ij_alp = sorted([k for k in alpha], key = lambda k : alpha[k])
    sorted_alp = [alpha[k] for k in ij_alp]
    # For each pair form the list of pairs it be trnsformed into
    pair2pair = {}
    for i in range(N):
        for j in range(i):
            alp = alpha[i,j]
            angle_tol = epsilon*(1/R[i] + 1/R[j])
            k = ij_alp.index((i,j))
            k1 = k
            while alp - sorted_alp[k1] < angle_tol:
                k1 -= 1
                if k1 == -1: break
            k1 += 1
            k2 = k
            while sorted_alp[k2] - alp < angle_tol:
                k2+=1
                if k2 == len(ij_alp): break
            pair2pair[i,j] = []
            pair2pair[j,i] = []
            for k in range(k1,k2):
                i1,j1 = ij_alp[k]
                if i1 in pt2pt[i] and j1 in pt2pt[j]:
                    pair2pair[i,j].append((i1,j1))
                    pair2pair[j,i].append((j1,i1))
                if i1 in pt2pt[j] and j1 in pt2pt[i]:
                    pair2pair[i,j].append((j1,i1))
                    pair2pair[j,i].append((i1,j1))

    t3 = time.process_time()
    print('Pair to pair list:   {:8.3f} sec CPU time'.format(t3-t2))

    # Find candidate permutations of points
    possible_subst = { i:pt2pt[i] for i in range(len(points)) }
    D0 = points[0].shape[0]
    projector = np.zeros((D0,D0))
    pers,ops =  match_next({}, possible_subst, badness_list, projector, points, dim, pair2pair, algo_fast, epsilon)

    t4 = time.process_time()
    print('\nApproximate symmetries search:   {:8.3f} sec CPU time'.format(t4-t3))

    return pers,ops

    
#---------------------------------------------------------------------------

def inclusive_closure(P,G):
    '''
    Driver routine to complete the group (perform the closure) and build its multiplication table
    P (IN, OUT)  - list of permutations isomorphic to G
                   permutations are specified as tuples, e.g. (0,2,1) means 0->0, 1->2, 2->1
    G (IN, OUT)  - group as a list of numpy matrices (arrays)
    return value - multiplication table as a dictionary (i,j):k meaning that G_i*G_j == G_k
    '''
    # Put the identity element to right place
    p0 = tuple(i for i in range(len(P[0])))
    i0 = P.index(p0)
    if i0 != 0:
        P[0], P[i0] = P[i0], P[0]
        G[0], G[i0] = G[i0], G[0]
    # Initialize multiplication table
    multab = {}
    multab[0,0] = 0
    absent = {}
    inew = 1
    iend = len(P)
    print("Inclusive closure & multiplication table build:")
    while inew < iend:
        t0 = time.process_time()
        t1 = 0e0
        for i in range(inew,iend):
            multab[i,0] = i
            multab[0,i] = i
            alrd_h = [False for k in range(len(P))]
            alrd_v = [False for k in range(len(P))]
            for j in range(1,i+1):
                Pij = tuple( P[i][p] for p in P[j] )
                fnd_h = False
                for k in range(len(P)):
                    if not alrd_h[k]:
                        if P[k] == Pij:
                            fnd_h = True
                            break
                if fnd_h:
                    multab[i,j] = k
                    alrd_h[k] = True
                    #print(i,j,' found ',k)
                else:
                    if Pij in absent:
                        absent[Pij].append((i,j))
                    else:
                        absent[Pij] = [(i,j)]
                    #print(i,j,' not found ',Pij)
                    #print(alrd_h)
                    #print(P)
                Pji = tuple( P[j][p] for p in P[i])
                fnd_v = False
                for k in range(len(P)):
                    if not alrd_v[k]:
                        if P[k] == Pji:
                            fnd_v = True
                            break
                if fnd_v:
                    multab[j,i] = k
                    alrd_v[k] = True
                    #print(j,i,' found_v ',k)
                else:
                    if Pji in absent:
                        absent[Pji].append((j,i))
                    else:
                        absent[Pji] = [(j,i)]
                    #print(j,i,' not found_v ',Pji)
                    #print(alrd_v)
                    #print(P)
            t2 = time.process_time()
            if t2-t1 > .3 or i == iend-1:
                print('\r', end='')
                print("Multiplication table: {:6.2f}% complete, {:8.3f} sec CPU time".
                      format(100*(i**2-inew**2 + 2*j + 1)/(iend**2-inew**2), t2 - t0),
                      end="", flush=True)
                t1 = t2
        inew = len(P)
        iend = inew
        for Pij in absent:
            P.append(Pij)
            for ij in absent[Pij]:
                multab[ij] = iend
            G.append(G[ij[0]].dot(G[ij[1]]))
            iend += 1
        t1 = time.process_time()
        print("\n{:10d} new elements found, {:8.3f} sec CPU time spent".format(len(absent),t1-t0))
        absent = {}
    print("Total group size now:{:7d}".format(len(P)))
    return multab    

# -----------------------------------------------------------------
#-----------------------------------------------------------------------------------
    

def exclusive_closure(P,G,points,epsilon):
    pass

#-----------------------------------------------------------------------------------

def np_int_zeros(n,m):
    return np.array([[0 for j in range(m)] for i in range(n)])

def np_extend(a,n,m):
    n1,m1 = a.shape
    am = np.concatenate((a,np_int_zeros(n1,m-m1)),axis=1)
    return np.concatenate((am,np_int_zeros(n-n1,m)),axis=0)

def np_index(arr,el):
    for i in range(len(arr)):
        c = arr[i]==el
        if type(c) == np.ndarray:
            c=c.all()
        if c:
            return i
    return -1


def np_inclusive_closure(P,G):
    # Put the identity element to right place
    N = len(P[0])
    p0 = np.arange(N)
    i0 = np_index(P,p0)
    if i0 != 0:
        P[i0], P[0] = P[0], p0
        g0 = np.copy(G[0])
        G[0], G[i0] = G[i0], g0
    # Initialize multiplication table
    multab = np_int_zeros(len(P),len(P))
    absent = {}
    inew = 1
    iend=len(P)
    print("Inclusive closure & multiplication table build:")
    while inew < iend:
        for i in range(inew,iend):
            multab[i,0] = i
            multab[0,i] = i
            alrd_h = np.array([False for k in range(len(P))])
            alrd_v = np.array([False for k in range(len(P))])
            for j in range(1,i+1):
                Pij = np.array( [P[i][p] for p in P[j]])
                k = np_index(P,Pij)
                if k>=0:
                    multab[i,j] = k
                    alrd_h[k] = True
                    #print(i,j,' found ',k)
                else:
                    pij = tuple(Pij)
                    if pij in absent:
                        absent[pij].append((i,j))
                    else:
                        absent[pij] = [(i,j)]
                    #print(i,j,' not found ',Pij)
                    #print(alrd_h)
                    #print(P)
                Pji = np.array([ P[j][p] for p in P[i]])
                k = np_index(P,Pji)
                if k>=0:
                    multab[j,i] = k
                    alrd_v[k] = True
                    #print(j,i,' found_v ',k)
                else:
                    pji = tuple(Pji)
                    if pji in absent:
                        absent[pji].append((j,i))
                    else:
                        absent[pji] = [(j,i)]
                    #print(j,i,' not found_v ',Pji)
                    #print(alrd_v)
                    #print(P)
            print('\r', end='')
            print("Multiplication table: {:6.2f}% complete".
                  format(100*(i**2-inew**2 + 2*j + 1)/(iend**2-inew**2)),
                  end="", flush=True)
        inew = len(P)
        iend = inew
        if absent != {}:
            Na = len(absent)
            P = np.concatenate((P,np_int_zeros(Na,N)),axis=0)
            #print(P)
            G = np.concatenate((G, np.zeros((Na,G[0].shape[0],G[0].shape[1]))),axis=0)
            #print(G)
            multab = np_extend(multab,len(P)+Na,len(P)+Na)
        for Pij in absent:
            P[iend] = Pij
            for ij in absent[Pij]:
                multab[ij] = iend
            G[iend] = G[ij[0]].dot(G[ij[1]])
            iend += 1
        print("\n{:10d} new elements found".format(len(absent)))
        absent = {}
    print("Total group size now:{:7d}".format(len(P)))
    return P,G,multab    

