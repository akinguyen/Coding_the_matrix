from vecutil import list2vec
from GF2 import *
import random
from HW import is_independent
from solver import solve
#from independence import is_independent
from matutil import rowdict2mat, coldict2mat, mat2rowdict, mat2coldict
from bitutil import str2bits, bits2mat, mat2bits, bits2str

a0 = list2vec([one,one,0,one,0,one])
b0 = list2vec([one,one,0,0,0,one])

def randGF2(): return random.randint(0,1)*one

def choose_secret_vector(s,t):
    check_random = False
    while not (check_random):
        u = list2vec([randGF2() for i in range(6)])
        if a0*u == s and b0*u == t:
            check_random = True
    return u

    
def generate_independent_vectors(n):
    B = []   
    while (len(B) < n*2):
        b = list2vec([randGF2() for i in range(6)])
        while b == a0 or b == b0:
            b = list2vec([randGF2() for i in range(6)])
        B.append(b)
        if not is_independent(B):
            del B[len(B)-1]
    return B


## String sharing secret

def string_matrix(words,nvectors):
    return bits2mat(str2bits(words),nvectors)

def decode_string_matrix(A):
    return bits2str(mat2bits(A))

def secret_matrix(n):
    A = [a0,b0]
    A.extend(generate_independent_vectors(n))
    return rowdict2mat(A)

def matrix_any_three(A,pairs):
    rowlist = mat2rowdict(A)
    return rowdict2mat([rowlist[i] for i in pairs])

def solve_matrix_matrix(A,B):
    b_col = mat2coldict(B)
    return coldict2mat([solve(A,b) for (i,b) in b_col.items()])
