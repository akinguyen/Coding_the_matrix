from vecutil import zero_vec,list2vec
from vec import Vec
from cancer_data import read_training_data
from matutil import mat2rowdict, mat2coldict, rowdict2mat, coldict2mat
from HW import QR_solve
def signum(u):
    v = zero_vec(u.D)
    for i in v.D:
        if u[i] >= 0:
            v[i] = 1
        else:
            v[i] = -1
    return v

def fraction_wrong(A,b,w):
    d = signum(A*w)
    c = [k for k in d.f if d.f[k] != b.f[k]]
    return len(c)/len(d.D)
    
    
def loss(A,b,w):
    u = (A*w-b)
    return u*u

def find_grad(A,b,w):
    return 2*(A*w-b)*A

def gradient_descent_step(A,b,w,sigma):
    return w - sigma*find_grad(A,b,w)

def gradient_descent(A,b,w,sigma,T):
    print_count = 1000
    count = 1
    for i in range(T):
        w = gradient_descent_step(A,b,w,sigma)
        if i == print_count:
            print(str(count)+'Loss function L(w) : ' + str(loss(A,b,w)))
            print('fraction wrong: ' + str(fraction_wrong(A,b,w)))
            print_count += 1000
            count += 1
    return w

def linear_regression_method(A,b):
    return QR_solve(A,b)


