from sympy import *
import numpy as np
import re

#define the conservative symbols
w1,w2,w3,w4 = symbols(('w1','w2','w3','w4'))
w1x,w2x,w3x,w4x= symbols(('w1x','w2x','w3x','w4x'))
w1y,w2y,w3y,w4y = symbols(('w1y','w2y','w3y','w4y'))
w1n,w2n,w3n,w4n = symbols(('w1n','w2n','w3n','w4n'))
w = Matrix([w1, w2, w3, w4])
wd = Matrix([[w1x,w2x,w3x,w4x],
            [w1y,w2y,w3y,w4y]])
n1, n2 = symbols(('n1', 'n2'))
n = [n1, n2]

#define reused quantities
g, mu, al,R = symbols(('g', 'mu', 'al','R'))
U2 = (w2**2 + w3**2)/(w1**2)
p = (g-1)*(w4 - w1*U2/2)
fE = (w4*g - (g-1)*w1*U2/2)

#define the 5*3 flux array
f = Matrix([[w2, w3],
           [w2*w2/w1 + p, w3*w2/w1],
           [w2*w3/w1, w3*w3/w1 + p],
           [w2*fE/w1, w3*fE/w1]])

#viscous flux
def tau(i, j):
    #i = grad dir, j = vel comp
    return mu*(w1*wd[i, j+1] - w[j+1]*wd[i,0])/w1**2

stress = np.zeros([2,2],dtype=object)
dev = 0
for i in range(0, 2):
    dev += 2*tau(i, i)/3
for i in range(0, 2):
    for j in range(0, 2):
        stress[i, j] = tau(i, j) + tau(j, i)
    stress[i, i] -= dev

fv = np.zeros([4, 2],dtype=object)
fv[1:3, :] = stress
for i in range(0, 2):
    for j in range(0, 2):
        fv[3, i] += w[j+1]*stress[j, i]/w1 - al*(w[j+1]*wd[i, j+1]/(w1**2) - w[j+1]**2*wd[i, 0]/w1**3)
    fv[3, i] += al*(w1*wd[i, 3]-w[3]*wd[i,0])/w1**2
#get the differential of the flux

A = np.zeros([2,4,4],dtype=object)
D = np.zeros([2,2,4,4], dtype=object)
for k in range(0, 4):
    for i in range(0, 4):
        for j in range(0,2):
            A[j, k, i] = diff(f[i,j]-fv[i,j], w[k])
            for m in range(0, 2):
                D[m,j,k,i] = diff(fv[i,j], wd[m, k])

#transform for python
zero,one=symbols(('zero', 'one'))
Ap = A.copy()
Ap[Ap==0] = zero
Ap[Ap==1] = one
Dp = D.copy()
Dp[Dp==0] = zero
Dp[Dp==1] = one

An = Matrix(A[0].T*n[0] + A[1].T*n[1])
Dn = Matrix((D[0,0].T*n[0] + D[0,1].T*n[1])*n[0] + (D[1,0].T*n[0] + D[1,1].T*n[1])*n[1])
#boundary conditions [p U T]
M = Matrix([[1,0,0,0],
          [w2/w1,w1,0,0],
          [w3/w1,0,w1,0],
          [U2/2,w2,w3,1/(g-1)]])

T = Matrix([[w1/p,0,0,-w1**2*R/p],
          [0,1,0,0],
          [0,0,1,0],
          [1,0,0,0]])
#solid wall, dU = 0. dT = 0
print "wall"
print "du/dn"
Dn = np.array(Dn.subs({w2:0,w3:0,n2**2:1-n1**2}).subs({n1**2:1-n2**2}))
for i in range(0,4):
    for j in range(0,4):
        Dn[i,j] = simplify(Dn[i,j])
print Dn
An = An*M*T
Dn = Dn*M*T
print "du"
print "w"
An = An.subs({w2:0,w3:0})
An = An.subs({w1x:w1n*n1,w2x:w2n*n1,w3x:w3n*n1,w4x:w4n*n1, w1y:w1n*n2,w2y:w2n*n2,w3y:w3n*n2,w4y:w4n*n2})
for i in range(0,4):
    for j in range(0,4):
        An[i,j] = simplify(An[i,j])
print An
print "dw/dn"
Dn = np.array(Dn.subs({w2:0,w3:0,n2:sqrt(1-n1**2)}))
for i in range(0,4):
    for j in range(0,4):
        Dn[i,j] = simplify(Dn[i,j])
print Dn
C = np.bmat([[An, -Dn],[-Dn, zeros([4,4])]])

print Matrix(C).rref()
#total pressure inlet + fixed temp
print "inlet"
#dpdu = -p*sqrt(U2)/(g*p/w1+(g-1)*U2/2)
#coeff = (-S[:,1]*n1 - S[:,2]*n2 + S[:,0]*dpdu).subs({w3:0,n1:-1,n2:0})
#print coeff



