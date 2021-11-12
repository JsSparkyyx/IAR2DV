import numpy as np
from NA import *

def InitDAO(U,dU,label):
    nA = np.zeros((U.shape[0]-dU.shape[0]+1,int(label.max())))
    nS = np.zeros((U.shape[0]-dU.shape[0]+1,U.shape[0]-dU.shape[0]+1))
    nM = np.zeros((U.shape[0]-dU.shape[0]+1,U.shape[0]-dU.shape[0]+1))
    nD = np.zeros((U.shape[0]-dU.shape[0]+1,U.shape[0]-dU.shape[0]+1))
    return nA,nS,nM,nD

def updSDAO1(i,j,k,dC,nA,S,nS,M,nM,nD,r):
    d = S[i+k-1,j+k-1]
    for l in range(dC.shape[1]):
        d -= (dC[i+k-1,l] - dC[j+k-1,l])**2
    if(d < 1e-15):
        d = 0
    nS[i,j] = d
    nS[j,i] = d
    d = d**0.5
    if(d <= r):
        nM[i,j] = 1
        nM[j,i] = 1
        if(M[i+k-1,j+k-1] == 0):
            nD[i,i] += 1
            nD[j,j] += 1
    else:
        nM[i,j] = 0
        nM[j,i] = 0

def updSDAO2(i,j,k,k2,dC,nA,S,nS,M,nM,nD,r):
    d = S[i+k-1,j+k2-1]
    for l in range(dC.shape[1]):
        d -= (dC[i+k-1,l] - dC[j+k2-1,l])**2
    if(d < 1e-15):
        d = 0
    nS[i,j] = d
    nS[j,i] = d
    d = d**0.5 
    if(d <= r):
        nM[i,j] = 1
        nM[j,i] = 1
        if(M[i+k-1,j+k2-1] == 0):
            nD[i,i] += 1
            nD[j,j] += 1
    else:
        nM[i,j] = 0
        nM[j,i] = 0
        
def updSDAO3(i,j,p,k,dC,nA,S,nS,M,nM,nD,r):
    d = S[i+k-1,j+p]
    for l in range(dC.shape[1]):
        d -= (dC[i+k-1,l] - dC[j+p,l])**2
    if(d < 1e-15):
        d = 0
    nS[i,j] = d
    nS[j,i] = d
    d = d**0.5 
    if(d <= r):
        nM[i,j] = 1
        nM[j,i] = 1
        if(M[i+k-1,j+p] == 0):
            nD[i,i] += 1
            nD[j,j] += 1
    else:
        nM[i,j] = 0
        nM[j,i] = 0

def updSDAO4(i,j,p,dC,nA,S,nS,M,nM,nD,r):
    d = S[i+p,j+p]
    for l in range(dC.shape[1]):
        d -= (dC[i+p,l] - dC[j+p,l])**2
    if(d < 1e-15):
        d = 0
    nS[i,j] = d
    nS[j,i] = d
    d = d**0.5 
    if(d <= r):
        nM[i,j] = 1
        nM[j,i] = 1
        if(M[i+p,j+p] == 0):
            nD[i,i] += 1
            nD[j,j] += 1
    else:
        nM[i,j] = 0
        nM[j,i] = 0

'''
    Run algorithm DAO
    U: objects of last round
    dU: indexes of deleted objects
    dC: deleted attributes
    A,S,M,D: matrix of last round
    r: radius
'''
def DAO(U,dU,dC,label,A,S,M,D,r):
    nA,nS,nM,nD = InitDAO(U,dU,label)
    for k in range(1,dU.shape[0]):
        for i in range(dU[k-1]-k+2,dU[k]-k+1):
            nD[i,i] += D[i+k-1,i+k-1]
            for k1 in range(1,dU.shape[0]):
                nD[i,i] -= M[i+k-1,dU[k1]]
            for j in range(int(label.max())):
                nA[i,j] = A[i+k-1,j]
            for j in range(i,dU[k]-k+1):
                updSDAO1(i,j,k,dC,nA,S,nS,M,nM,nD,r)
            for k2 in range(k+1,dU.shape[0]):
                for j in range(dU[k2-1]-k2+2,dU[k2]-k2+1):
                    updSDAO2(i,j,k,k2,dC,nA,S,nS,M,nM,nD,r)
            for j in range(dU[-1]-dU.shape[0]+2,U.shape[0]-dU.shape[0]+1):
                updSDAO3(i,j,dU.shape[0]-1,k,dC,nA,S,nS,M,nM,nD,r)
    for i in range(dU[-1]-dU.shape[0]+2,U.shape[0]-dU.shape[0]+1):
        nD[i,i] += D[i+dU.shape[0]-1,i+dU.shape[0]-1]
        for k1 in range(1,dU.shape[0]):
            nD[i,i] -= M[i+dU.shape[0]-1,dU[k1]]
        for j in range(int(label.max())):
            nA[i,j] = A[i+dU.shape[0]-1,j]
        for j in range(i,U.shape[0]-dU.shape[0]+1):
            updSDAO4(i,j,dU.shape[0]-1,dC,nA,S,nS,M,nM,nD,r)
    nG = getInverseDiaMatrix(nD)
    nI = nM.dot(nA)
    nH = np.dot(nG,nI)
    Q = getQuality(nH)
    return Q,nA,nS,nM,nD