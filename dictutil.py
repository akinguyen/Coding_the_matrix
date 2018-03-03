# Copyright 2013 Philip N. Klein
def dict2list(dct, keylist): return [ dct[i] for i in keylist ]
def list2dict(L, keylist): return { keylist[i]:L[i] for i in range(len(L)) }
