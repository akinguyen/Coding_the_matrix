from factoring_support import intsqrt, dumb_factor, primes, prod
from math import sqrt
from GF2 import *
from vec import Vec
from echelon import transformation_rows
from matutil import rowdict2mat, coldict2mat, mat2rowdict, mat2coldict

def root_method(N):
    a = intsqrt(N)+2
    while not ((a**2 - N) == intsqrt(a**2 - N)**2):
        a += 1
    b = sqrt(a**2 - N)
    return a - b

def gcd(x,y): return x if y == 0 else gcd(y,x%y)

def int2GF2(x):
    if x%2 == 0:
        return 0
    return one

def make_Vec(primeset,factors):
    return Vec(primeset,{a[0]:int2GF2(a[1]) for a in factors})

def find_candidates(N,primeset):
    roots = []
    rowlist = []
    x = intsqrt(N)+2
    while len(roots) < len(primeset)+1:
        factors = dumb_factor(x*x - N,primeset)
        if len(factors) > 0:
            roots.append(x)
            rowlist.append(make_Vec(primeset,factors))
        x += 1
    return roots,rowlist

def find_a_and_b(v,roots,N):
    alist = [roots[i] for i in v.D if v[i] == one]
    a = prod(alist)
    c = prod([x*x - N for x in alist])
    b = intsqrt(c)
    assert b*b == c
    return (a,b)


""" For kN = (a-b)(a+b) , which a^2 - b^2 is divisible by N,
    implies that the primes in k's and N's prime bags will
    belong to either (a-b)'s or (a+b)'s prime bags.
    Thus, assume N is a product of two primes, p and q, they will
    also belong either to (a-b)'s or (a+b)'s prime bags.
    By finding gcd of a-b and a+b, we can discover p and q
    Ex: a-b = p*t and p = p*1 ==> gcd(a-b,p) == p
        a+b = q*s and q = q*1 ==> gcd(a+b,q) == q """

def factor(N,primeset):
    roots, rowlist = find_candidates(N,primeset)
    M = transformation_rows(rowlist)
    m = rowdict2mat(M)
    a = rowdict2mat(rowlist)
    ma = mat2rowdict(m*a)
    zero_pos = [i for i in ma if ma[i] == 0*ma[0]]
    
    for i in zero_pos:
        a,b = find_a_and_b(M[i],roots,N)
        factor1,factor2 = gcd(a+b,N),gcd(a-b,N)
        if factor1*factor2 == N:
            return factor1,factor2


    
