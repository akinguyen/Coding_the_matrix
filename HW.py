from solver import solve
from mat import Mat
from matutil import rowdict2mat, coldict2mat, mat2coldict, mat2rowdict, listlist2mat,identity
from vec import Vec
from vecutil import list2vec, zero_vec
from independence import rank
from GF2 import *
from triangular import triangular_solve
from echelon import transformation
from QR import factor
from triangular import triangular_solve
from read_data import read_vectors
from cancer_data import read_training_data
from math import sqrt
#HW chapter 5
def rep2vec(u,veclist):
    return coldict2mat(veclist)*u

def vec2rep(veclist,v):
    return solve(coldict2mat(veclist),v)

def is_superfluous(L,i):
    if len(L) <= 1:
        return False
    L_copy = L.copy()
    check_vec =  L_copy.pop(i)
    A = coldict2mat(L_copy)
    res = check_vec - A*solve(A,check_vec)
    return res*res < 10e-14
    
def is_independent(L):
    for i in range(len(L)):
        if(is_superfluous(L,i)):
            return False
        return True

def subset_basis(T):
    B = []
    for i in range(len(T)):
        B.append(T[i])
        if not is_independent(B):
            B.remove(T[i])
    return B

def exchange(S,A,z):
    S_new = S.copy()
    S_new.append(z)
    for i in range(len(S_new)):
        if is_superfluous(S_new, i) and S_new[i] not in A and S_new[i] is not z:
            return S_new[i]

#HW chapter 6
def morph(S,B):
    pairs = Ax = b
    S_copy = S.copy()
    B_copy = []
    for i in range(len(B)):
        w = exchange(S_copy,B_copy,B[i])
        pairs.append((B[i],w))
        S_copy.append(B[i])
        S_copy.remove(w)
        B_copy.append(B[i])
    return pairs

def my_is_independent(L):
    return len(L) == rank(L)

def my_rank(L):
    return len(subset_basis(L))

def direct_sum_decompose(U,V,w):
    U_copy = U.copy()
    U_copy.extend(V)
    linear_comb = solve(coldict2mat(U_copy),w)
    u_linear_comb = list2vec([linear_comb[i] for i in range(len(U))])
    v_linear_comb = list2vec([linear_comb[i] for i in range(len(U),len(V)+len(U))])
    return (coldict2mat(U)*u_linear_comb,coldict2mat(V)*v_linear_comb)
    
def is_invertible(M):
    return len(mat2coldict(M)) == len(mat2rowdict(M)) and my_is_independent([ v for (u,v) in mat2coldict(M).items()])

def find_matrix_inverse(A):
    return coldict2mat([solve(A,b) for (a,b) in mat2coldict(identity(A.D[0],one)).items()])

#HW chapter 7
def is_echelon(rowlist):
    k = -2
    for i in range(len(rowlist)):
        for j in range(len(rowlist[0].D)):
            if rowlist[i][j] is not 0:
                if j <= k:
                    return False
                k = j
                break
    return True

def remove_irr_col(rowlist):
    entry_col = []
    label_row = range(len(rowlist))
    label_col = list(rowlist[0].D)
    assert is_echelon(rowlist)
    for i in label_row:
        for j in label_col:
            if rowlist[i][j] is not 0:
                entry_col.append(j)
                break
            
    remove_col = [i for i in label_col if i not in entry_col]
    collist = mat2coldict(rowdict2mat(rowlist)) 
    for i in remove_col:
        collist.pop(i)
    return collist


def new_tria_solve(rowlist,b):
    collist = remove_irr_col(rowlist)
    new_rowlist = [ values for (keys,values) in mat2rowdict(coldict2mat(collist)).items()]
    return triangular_solve(new_rowlist,list(new_rowlist[0].D),b)

def has_solution(rowlist,b):
    zero_vector = zero_vec(rowlist[0].D)
    for i in range(len(rowlist)):
        if (rowlist[i] == zero_vector and b[i] is not 0):
            return False
    return True

def echelon_solve(rowlist,label_list,b):
    x = zero_vec(label_list)
    entry_col = []
    label_row = range(len(rowlist))
    assert is_echelon(rowlist)
    for i in label_row:
        for j in label_list:
            if rowlist[i][j] is not 0:
                entry_col.append(j)
                break
    for i in reversed(label_row):
        for c in reversed(entry_col):
            row = rowlist[i]
            x[c] = (b[i] - x*row)/row[c]
            entry_col.remove(c)
            break
    return x


def solve2(A,b):
    M = transformation(A)
    U = M*A
    U_rowdict = mat2rowdict(U)
    rowlist = [U_rowdict[i] for i in sorted(U_rowdict)]
    label_list = (A.D[1])
    return echelon_solve(rowlist,label_list,M*b)


""" basis of NS(A.transpose) """    
def basis_AT(A):
    basis = []
    M = transformation(A)
    U = M*A
    M_rowdict = mat2rowdict(M)
    U_rowdict = mat2rowdict(U)
    M_rowlist = [M_rowdict[i] for i in sorted(M_rowdict)]
    U_rowlist = [U_rowdict[i] for i in sorted(U_rowdict)]
    zero_vector = zero_vec(U_rowlist[0].D)
    for i in range(len(U_rowlist)):
        if (U_rowlist[i] == zero_vector):
            basis.append(M_rowlist[i])
    return basis
            
#HW chapter 9
def find_null_basis(A):
    Q,R = factor(A)
    R_inverse = find_matrix_inverse(R)
    R_inverse_list = mat2coldict(R_inverse)
    Q_list = mat2coldict(Q)
    zero_vector = zero_vec(Q_list[0].D)
    return [R_inverse_list[i] for i in range(len(Q_list)) if Q_list[i] is zero_vector ]

    
def QR_solve(A,b):
    Q,R = factor(A)
    R_rowlist = mat2rowdict(R)
    label_list = sorted(A.D[1],key =repr)
    return triangular_solve(R_rowlist,label_list,Q.transpose()*b)

""" Linear Regression """
def linear_regression(data):
    datalist = read_vectors(data)
    x = list(datalist[0].D)[1]
    y = list(datalist[0].D)[0]
    x_domain = {1,x}
    x_rowlist = []
    y_list = []
    for v in datalist:
        x_rowlist.append(Vec(x_domain,{1:1, x:v[x]}))
        y_list.append(v[y])
    y_vec = list2vec(y_list)
    minimize = QR_solve(rowdict2mat(x_rowlist),y_vec)
    return minimize[1],minimize[x] # return b,a 

#HW chapter 11
def squared_Frob(A):
    rowlist = mat2rowdict(A)
    return sum([element*element for (i,element) in rowlist.items()])

def SVD_solve(A,b):
    U,S,V = svd.factor(A)
    return V*find_matrix_inverse(S)*U.transpose()*b

#HW chapter 12
def power_method(A, k):
    v = Vec(A.D[1],{c:1 for c in A.D[1]})
    for i in range(k):
            w = A*v
            v = w/sqrt(w*w)
            lambda1 = v*A*v
    return v, lambda1

def inverse_power_method(A,k):
    v = Vec(A.D[1],{c:1 for c in A.D[1]})
    for i in range(k):
            w = solve(A,v)
            v = w/sqrt(w*w)
            lambda1 = v*A*v
    return v, lambda1

def find_eigen_value(A,eigenvector):
    return A*eigenvector*eigenvector/(eigenvector*eigenvector)

def find_eigenvector(A,eigenvalue):
    A_eigenvalue = A - eigenvalue*identity(A.D)
    return find_null_basis(A_eigenvalue)


