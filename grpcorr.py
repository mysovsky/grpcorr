#GRPCORR
#Copyright (C) 2019 Andrey S. Mysovsky, Institute of Geochemistry SB RAS

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from math import sqrt
import random
import scipy.linalg as sla

def mpow(M,p):
    ''' M - symmetric positively defined matrix (IN)
        p - real number (IN)
        return value - M^p 
    '''
    N = M.shape[0]
    e,v = np.linalg.eig(M)
    res = np.zeros(shape = (N,N))
    for i in range(N):
        res += np.outer(v[:,i],v[:,i])*pow(e[i],p)
    return res


def make_unitary(U):
    S = np.matmul(U,U.transpose())
    return np.matmul(sla.fractional_matrix_power(S,-0.5),U)


def make_F(G,mtab):
    ''' G   ( IN) - group as a list of numpy matricies
        mtab (IN)  - group multiplication table; 
        mtab[i][j] is the number of element G[i]*G[j]
        return value - array of F[i,j] = G[i]*G[j] - G[(ij)]
    '''
    N = len(G)
    return [[np.matmul(G[i],G[j]) - G[mtab[i][j]] for j in range(N)] for i in range(N)]

def grp_error(F):
    ''' F  - array of F[i,j] = G[i]*G[j] - G[(ij)]
    '''
    s = 0e0
    for ff in F:
        for f in ff:
            s += np.linalg.norm(f)**2
    return sqrt(s)

def grp_rotate(G,R):
    for i in range(len(G)):
        G[i] = np.matmul(sla.expm(-R), np.matmul(G[i], sla.expm(R)))

        
def mtab_correction(G,F):
    ''' perform multiplication table based group correction (one iteration)
        G (IN,OUT) - group as a list of numpy matricies
        F (IN) - array of F[i,j] = G[i]*G[j] - G[(ij)]
    '''
    N = len(G)
      # The group size
    Ginv = [np.linalg.inv(G[i]) for i in range(N)]
      # inverse matricies
    for i in range(N):
        D = np.zeros(shape = G[0].shape)
        for j in range(N):
            D += np.matmul(Ginv[j], F[j][i]) + np.matmul(F[i][j], Ginv[j])
        D /= 2*N
        G[i] -= D


def multab_group_correction(G,mtab, eps = 1e-6 ,maxit = 100 ):
    ''' perform multiplication table based group correction until the 
        convergence is achieved
        G (IN,OUT) - group as a list of numpy matricies
        mtab (IN)  - group multiplication table; 
        mtab[i][j] is the number of element G[i]*G[j]
        eps  (IN, real) - the convergence threshold for total deviation norm 
                          from multiplication table
        maxit (IN, int) - maximum number of iterations
    '''
    it = 0
    N = len(G)
    while it < maxit:
        for i in range(N):
            G[i]=make_unitary(G[i])
        F = make_F(G,mtab)
        # calculate the error
        errval = grp_error(F)
        print('simple_grp_correct: iteration {:4d} error {:15.8f}'.format(it,errval))
        if errval < eps:
            print("Convergence is achieved!")
            break
        mtab_correction(G,F)
        it += 1
    if it>=maxit:
        print("Maximum number of iterations reached, the fit is not converged!!!")


def make_Q(a,b):
    '''
    a (IN) - list of list of numpy vectors
    b (IN) - list of list numpy vectors
    return value - list of numpy matricies
        Q[i] = SUM(j) |a[i][j]><b[i][j]|
    '''
    Q = []
    for i in range(len(a)):
        s = np.zeros(shape = (a[i][0].shape[0],b[i].shape[0]))
        for j in range(len(a[i])):
            s += np.outer(a[i][j],b[i][j])
        Q.append(s)
    return Q

def kronecker_prod(A,B):
    na1, na2 = A.shape
    nb1, nb2 = B.shape
    n1 = na1*nb1
    n2 = na2*nb2
    res = np.zeros(shape = (n1,n2))
    for i in range(na1):
        for j in range(na2):
            res[i*nb1:(i+1)*nb1, j*nb2:(j+1)*nb2] = A[i,j]*B
    return res

def mtr_to_vec(A):
    res = np.zeros(shape=(A.shape[0]*A.shape[1]))
    k=0
    for j in range(A.shape[1]):
        for i in range(A.shape[0]):
            res[k] = A[i][j]
            k += 1
    return res

def vec_to_mtr(v,n,m):
    res = np.zeros(shape = (n,m))
    k = 0
    for j in range(m):
        for i in range(n):
            res[i][j] = v[k]
            k += 1
    return res

def make_L0(G):
    N = len(G)
    In = np.identity(G[0].shape[0])
    L = 4*N*kronecker_prod(In,In)
    for i in range(N):
        L -= 2*kronecker_prod(G[i],G[i])
        L -= 2*kronecker_prod(G[i].transpose(),G[i].transpose())
    return L
        
def make_L1(G,Q):
    N = len(G)
    GQ = np.zeros(shape = G[0].shape)
    for i in range(N):
        GQ += np.matmul(G[i], Q[i].transpose())*0.5
        GQ += np.matmul(Q[i], G[i].transpose())*0.5
        GQ += np.matmul(G[i].transpose(), Q[i])*0.5
        GQ += np.matmul(Q[i].transpose(), G[i])*0.5
    IN = np.identity(G[0].shape[0])
    L  = kronecker_prod(IN,GQ)
    L += kronecker_prod(GQ.transpose(),IN)
    for i in range(N):
        L -= kronecker_prod(Q[i],G[i])
        L -= kronecker_prod(G[i],Q[i])
        L -= kronecker_prod(G[i].transpose(), Q[i].transpose())
        L -= kronecker_prod(Q[i].transpose(), G[i].transpose())
    return L

def grad_R(G,Q):
    N = len(G)
    g = np.zeros(shape = G[0].shape)
    for i in range(N):
        g += np.matmul(G[i], Q[i].transpose()) - np.matmul(Q[i].transpose(), G[i])
        g += np.matmul(G[i].transpose(), Q[i]) - np.matmul(Q[i], G[i].transpose())
    return g

def q_correction(G,Q):
    d = G[0].shape[0]
    L = make_L0(G)
    g = grad_R(G,Q)
    e,v = np.linalg.eig(L)
    r = np.zeros(shape=(d**2))
    for i in range(d**2):
        c = v[:,i].dot(mtr_to_vec(g))
        if abs(c) < 1e-6:
            continue
        r += np.real(v[:,i]*c/e[i])
    return vec_to_mtr(r,d,d)

def q_simple_correction(G,Q):
    return grad_R(G,Q)/(4*len(G))

def grp_diff(G,Q):
    S = 0e0
    for i in range(len(G)):
        S += np.linalg.norm(G[i]-Q[i])**2
    return sqrt(S)

def lsf_group_correction(G,Q,mtab,maxit=100, eps_mtab=1e-6, eps_diff=1e-6, eps_grad = 1e-6):
    ''' perform multiplication table based group correction with 
        simultaneous leat squares fit until the 
        convergence is achieved
        G (IN,OUT) - group as a list of numpy matricies
        Q (IN)     - list of numpy matricies Q[i] to which G[i] must be as close as possible
        mtab (IN)  - group multiplication table; 
        mtab[i][j] is the number of element G[i]*G[j]
        maxit (IN, int) - maximum number of iterations
        eps_mtab  (IN, real) - the convergence threshold for multiplication table violation
        eps_diff  (IN, real) - the convergence threshold for total difference between G and Q
        eps_frad  (IN, real) - the convergence threshold for Sum. sqs. gradient
    '''
    it = 0
    N = len(G)
    err_diff_prev = 0e0
    while it < maxit:
        for i in range(N):
            G[i]=make_unitary(G[i])
        convg = True
        F = make_F(G,mtab)
        # calculate the multiplication table violation value
        err_mtab = grp_error(F)
        if err_mtab > eps_mtab:
            mtab_correction(G,F)
            convg = False
        # calculate the Sum. sqs for G and Q difference
        R = q_simple_correction(G,Q)
        grp_rotate(G,R)
        err_diff = grp_diff(G,Q)
        delta_diff = err_diff_prev - err_diff
        err_grad = np.linalg.norm(R)
        if abs(delta_diff) > eps_diff or  err_grad > eps_grad:
            convg = False
        print('it = {:4d} multab violation = {:12.8f} norm(G-Q) = {:12.8f} Delta = {:12.8f} norm(R) = {:12.8f}'.\
              format(it,err_mtab,err_diff,delta_diff,err_grad))
        if convg:
            print('Convergence achieved!!!')
            break
        it += 1
        err_diff_prev = err_diff
    if it>=maxit:
        print("Maximum number of iterations reached, the fit is not converged!!!")

def rnd_mtr(N):
    A = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            A[i,j] = 2*random.random()-1e0
    return A

def grp_shake(G,eps):
    ''' shake the group by adding random real numbers to all matrix elements
    '''
    for i in range(len(G)):
        for j in range(G[i].shape[0]):
            for k in range(G[i].shape[1]):
                G[i][j,k] += eps*(2*random.random()-1e0)
            
def m3d_to_np(M):
    return np.array([[M[i,j] for j in [0,1,2]] for i in [0,1,2]])

