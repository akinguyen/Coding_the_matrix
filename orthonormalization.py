# Copyright 2013 Philip N. Klein
from vec import Vec
from vecutil import list2vec
from orthogonalization import project_along, project_orthogonal, orthogonalize, aug_orthogonalize
from math import sqrt
from matutil import coldict2mat, rowdict2mat, mat2rowdict

def project_orthogonal2(b,vlist):
    if vlist == []:
        return b
    else:
        b = b - project_along(b,vlist[len(vlist)-1])
        vlist.remove(vlist[len(vlist)-1])
        return project_orthogonal2(b,vlist)

def orthonormalize(L):
    assert isinstance(L,list)
    L_star = orthogonalize(L)
    return [vstar/sqrt(vstar*vstar) for vstar in L_star]
           

def aug_orthonormalize(L):
    def adjust(v,mul):
        return Vec(v.D,{i:mul[i]*v[i] for i in v.D})
    
    assert isinstance(L,list)
    Qstar,Rstar = aug_orthogonalize(L)
    mul = [sqrt(vstar*vstar) for vstar in Qstar]
    Qlist = [Qstar[i]/mul[i] for i in range(len(mul))]
    Rlist = [adjust(r,mul) for r in Rstar]
    return Qlist,Rlist
