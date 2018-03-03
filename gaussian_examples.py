from mat import Mat
from GF2 import one

Ad = (set([1,2,3,4]), set(['A','B','C','D']))
Af = {k:one for k in {(2,'A'),(2,'C'),(2,'D'),(1,'C'),(1,'D'),(3,'A'),(3,'D'),(4,'A'),(4,'B'),(4,'C'),(4,'D')}}
A = Mat(Ad,Af)

Bd = (set([1,2,3,4]), set(['A','B','C','D']))
Bf = {k:one for k in {(1,'A'),(1,'B'),(2,'A'),(2,'C'),(3,'B'),(3,'C'),(3,'D'),(4,'A')}}
B=Mat(Bd, Bf)
