# GRPCORR - module implementing finite matrix group corrections

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

from math import sqrt
import random
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

def make_unitary(U):
    S = U*U.T
    return sla.fractional_matrix_power(S,-0.5)*U


def make_F(G,mtab):
    ''' G   ( IN) - group as a list of numpy matricies
        mtab (IN)  - group multiplication table; 
        mtab[i][j] is the number of element G[i]*G[j]
        return value - array of F[i,j] = G[i]*G[j] - G[(ij)]
    '''
    N = len(G)
    return [[ G[i]*G[j] - G[mtab[i][j]] for j in range(N)] for i in range(N)]

def grp_error(F):
    ''' F  - array of F[i,j] = G[i]*G[j] - G[(ij)]
    '''
    s = 0e0
    for ff in F:
        for f in ff:
            s += la.norm(f)**2
    return sqrt(s)

def grp_rotate(G,R):
    for i in range(len(G)):
        G[i] = sla.expm(-R)*G[i]*sla.expm(R)

        
def mtab_correction(G,F):
    ''' perform multiplication table based group correction (one iteration)
        G (IN,OUT) - group as a list of numpy matricies
        F (IN) - array of F[i,j] = G[i]*G[j] - G[(ij)]
    '''
    N = len(G)
      # The group size
    Ginv = [la.inv(G[i]) for i in range(N)]
      # inverse matricies
    for i in range(N):
        D = np.asmatrix(np.zeros(shape = G[0].shape))
        for j in range(N):
            D = D + Ginv[j]*F[j][i] + F[i][j]*Ginv[j]
        D /= 2*N
        G[i] = G[i] - D


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
#            pass
            G[i]=make_unitary(G[i])
        F = make_F(G,mtab)
        # calculate the error
        errval = grp_error(F)
        print('simple_grp_correct: iteration {:4d} error {:25.18f}'.format(it,errval))
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
        s = np.zeros(shape = (a[i][0].shape[0],b[i][0].shape[0]))
        for j in range(len(a[i])):
            s = s + np.outer(a[i][j],b[i][j])
        Q.append(np.asmatrix(s))
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
    return np.asmatrix(res)

def mtr_to_vec(A):
    res = np.zeros(shape=(A.shape[0]*A.shape[1]))
    k=0
    for j in range(A.shape[1]):
        for i in range(A.shape[0]):
            res[k] = A[i,j]
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
    In = np.asmatrix(np.identity(G[0].shape[0]))
    L = 4*N*kronecker_prod(In,In)
    for i in range(N):
        L -= 2*kronecker_prod(G[i],G[i])
        L -= 2*kronecker_prod(G[i].transpose(),G[i].transpose())
    return L
        
def make_L1(G,Q):
    N = len(G)
    GQ = np.zeros(shape = G[0].shape)
    for i in range(N):
        GQ = GQ + ( G[i]*Q[i].H + Q[i]*G[i].H + G[i].H*Q[i] + Q[i].H*G[i] )*0.5
    IN = np.asmatrix(np.identity(G[0].shape[0]))
    L  = kronecker_prod(IN,GQ)
    L += kronecker_prod(GQ.T,IN)
    for i in range(N):
        L -= kronecker_prod(Q[i].conj(), G[i])
        L -= kronecker_prod(G[i].conj(), Q[i])
        L -= kronecker_prod(G[i].T, Q[i].H)
        L -= kronecker_prod(Q[i].T, G[i].H)
    return L

def grad_R(G,Q):
    N = len(G)
    g = np.zeros(shape = G[0].shape)
    for i in range(N):
        #debug
        #print(la.norm(np.imag(G[i])), la.norm(np.imag(Q[i])))                    
        g = g + G[i]*Q[i].H - Q[i].H*G[i] + G[i].H*Q[i] - Q[i]*G[i].H
    return g

def q_correction(G,Q):
    d = G[0].shape[0]
    L = make_L0(G)
    g = grad_R(G,Q)
    e,v = la.eig(L)
    r = np.zeros(shape=(d**2))
    for i in range(d**2):
        c = np.asarray(v)[:,i].dot(mtr_to_vec(g))
        if abs(c) < 1e-6:
            continue
        r = r + np.asarray(v)[:,i]*c/e[i]
    return vec_to_mtr(r,d,d)

def q_simple_correction(G,Q):
    return grad_R(G,Q)/(4*len(G))

def grp_diff(G,Q):
    S = 0e0
    for i in range(len(G)):
        S += la.norm(G[i]-Q[i])**2
    return sqrt(S)

def ab_diff(G,a,b):
    S = 0e0
    for i in range(len(G)):
        for j in range(len(a[i])):
            S += la.norm(np.matmul(G[i],a[i][j]) - b[i][j])**2
    return sqrt(S)
        
# ---------------------------------------------------------------------

def lsf_group_correction(G,Q,mtab, algo=0, maxit=100,
                         eps_mtab=1e-6, eps_diff=1e-6, eps_grad = 1e-6):
    ''' perform multiplication table based group correction with 
        simultaneous leat squares fit until the 
        convergence is achieved
        G (IN,OUT) - group as a list of numpy matricies
        Q (IN)     - list of numpy matricies Q[i] to which G[i] must be as close as possible
        mtab (IN)  - group multiplication table; 
        mtab[i][j] is the number of element G[i]*G[j]
        algo  (IN, int) - selector of the q-correction formula
               algo = 0 - simplified formula
               algo = 1 - supermatrix based solution
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
        if algo == 0:
            R = q_simple_correction(G,Q)
        elif algo == 1:
            R = q_correction(G,Q)
        else:
            raise IndexError('lsf_group_correction: algo parameter must be 0 or 1')
        grp_rotate(G,R)
        err_diff = grp_diff(G,Q)
        delta_diff = err_diff_prev - err_diff
        err_grad = la.norm(R)
        if abs(delta_diff) > eps_diff or  err_grad > eps_grad:
            convg = False
        print('it = {:4d} multab violation = {:25.18f} norm(G-Q) = {:25.18f} Delta = {:25.18f} norm(R) = {:25.18f}'.\
              format(it,err_mtab,err_diff,delta_diff,err_grad))
        if convg:
            print('Convergence achieved!!!')
            break
        it += 1
        err_diff_prev = err_diff
    if it>=maxit:
        print("Maximum number of iterations reached, the fit is not converged!!!")

# -----------------------------------------------------------------------------------

def abfit_group_correction(G, a, b, mtab, algo=0, maxit=100,
                         eps_mtab=1e-6, eps_diff=1e-6, eps_grad = 1e-6):
    ''' perform multiplication table based group correction with 
        simultaneous leat squares fit until the 
        convergence is achieved
        G (IN,OUT) - group as a list of numpy matricies
        a, b (IN)  - two lists of numpy vectors
               abfit is trying to make G[i]*a[i][j] = b[i][j] with the best 
               availiable accuracy
        mtab (IN)  - group multiplication table; 
        mtab[i][j] is the number of element G[i]*G[j]
        algo  (IN, int) - selector of the q-correction formula
               algo = 0 - simplified formula
               algo = 1 - supermatrix based solution
        maxit (IN, int) - maximum number of iterations
        eps_mtab  (IN, real) - the convergence threshold for multiplication table violation
        eps_diff  (IN, real) - the convergence threshold for total difference between G and Q
        eps_frad  (IN, real) - the convergence threshold for Sum. sqs. gradient
    '''
    it = 0
    N = len(G)
    err_diff_prev = 0e0
    Q = make_Q(a,b)
    for i in range(N):
        Q[i] = make_unitary(Q[i])
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
        if algo == 0:
            R = q_simple_correction(G,Q)
        elif algo == 1:
            R = q_correction(G,Q)
        else:
            raise IndexError('lsf_group_correction: algo parameter must be 0 or 1')
        grp_rotate(G,R)
        err_diff = ab_diff(G,a,b)
        delta_diff = err_diff_prev - err_diff
        err_grad = la.norm(R)
        if abs(delta_diff) > eps_diff or  err_grad > eps_grad:
            convg = False
        print('it = {:4d} multab error = {:25.21f} |G*a-b| = {:25.21f} Delta = {:25.21f} norm(R) = {:25.21f}'.\
              format(it,err_mtab,err_diff,delta_diff,err_grad))
        if convg:
            print('Convergence achieved!!!')
            break
        it += 1
        err_diff_prev = err_diff
    if it>=maxit:
        print("Maximum number of iterations reached, the fit is not converged!!!")

# ----------------------------------------------------------------------------------
        
def rnd_mtr(N):
    A = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            A[i,j] = 2*random.random()-1e0
    return np.asmatrix(A)

def grp_shake(G,eps):
    ''' shake the group by adding random real numbers to all matrix elements
    '''
    for i in range(len(G)):
        for j in range(G[i].shape[0]):
            for k in range(G[i].shape[1]):
                G[i][j,k] += eps*(2*random.random()-1e0)
            
def m3d_to_np(M):
    return np.matrix([[[M[i,j] for j in [0,1,2]] for i in [0,1,2]]])

