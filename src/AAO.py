import numpy as np
from NA import *

def InitAAO(U,dU,label):
    nA = np.zeros((U.shape[0]+dU.shape[0],int(label.max())))
    nS = np.zeros((U.shape[0]+dU.shape[0],U.shape[0]+dU.shape[0]))
    nM = np.zeros((U.shape[0]+dU.shape[0],U.shape[0]+dU.shape[0]))
    nD = np.zeros((U.shape[0]+dU.shape[0],U.shape[0]+dU.shape[0]))
    return nA,nS,nM,nD

'''
    update area ┍ *  *  *      ┑
                |    *  *      |
                |       *      |
                |              |
                |              |
                ┕              ┙
'''
def updSAAO1(i,j,dC,nA,S,nS,M,nM,nD,r):
    d = S[i,j]
    if(dC.ndim != 1):
        for k in range(dC.shape[1]):
            d += (dC[i,k] - dC[j,k])**2
    else:
        d += (dC[i] - dC[j])**2
    nS[i,j] = d
    nS[j,i] = d
    d = d**0.5 
    if(d <= r):
        nM[i,j] = 1
        nM[j,i] = 1
    else:
        nM[i,j] = 0
        nM[j,i] = 0
        if(M[i,j] == 1):
            if(i != j):
                nD[i,i] -= 1
                nD[j,j] -= 1
            else:
                nD[i,i] -= 1

'''
    update area ┍         * * *┑
                |         * * *|
                |         * * *|
                |              |
                |              |
                ┕              ┙
'''
def updSAAO2(i,j,nU,nA,S,nS,nM,nD,r):
    d = 0
    for k in range(nU.shape[1]):
        d += (nU[i,k] - nU[j,k])**2
    nS[i,j] = d
    nS[j,i] = d
    d = d**0.5 
    if(d <= r):
        nM[i,j] = 1
        nM[j,i] = 1
        if(i != j):
            nD[i,i] += 1
            nD[j,j] += 1
        else:
            nD[i,i] += 1

'''
    update area ┍              ┑
                |              |
                |              |
                |         * * *|
                |           * *|
                ┕             *┙
'''
def updSAAO3(i,j,nU,nA,S,nS,nM,nD,r):
    d = 0
    for k in range(nU.shape[1]):
        d += (nU[i,k] - nU[j,k])**2
    nS[i,j] = d
    nS[j,i] = d
    d = d**0.5 
    if(d <= r):
        nM[i,j] = 1
        nM[j,i] = 1
        if(i != j):
            nD[i,i] += 1
            nD[j,j] += 1
        else:
            nD[i,i] += 1

'''
    Run algorithm AAO
    U: objects of last round
    dU: new objects
    nU: all objects
    dC: new attributes
    A,S,M,D: matrix of last round
    r: radius
'''
def AAO(U,dU,nU,dC,label,A,S,M,D,r):
    nA,nS,nM,nD = InitAAO(U,dU,label)
    # print(nA.shape,A.shape)
    for i in range(U.shape[0]):
        nD[i,i] += D[i,i]
        for j in range(int(label.max())):
            nA[i,j] = A[i,j]
        for j in range(i,U.shape[0]):
            updSAAO1(i,j,dC,nA,S,nS,M,nM,nD,r)
        for j in range(U.shape[0],U.shape[0]+dU.shape[0]):
            updSAAO2(i,j,nU,nA,S,nS,nM,nD,r)
    for i in range(U.shape[0],U.shape[0]+dU.shape[0]):
        for j in range(int(label.max())):
            if(label[i] == j + 1):
                nA[i,j] = 1
        for j in range(i,U.shape[0]+dU.shape[0]):
            updSAAO3(i,j,nU,nA,S,nS,nM,nD,r)
    nG = getInverseDiaMatrix(nD)
    nI = nM.dot(nA)
    nH = np.dot(nG,nI)
    Q = getQuality(nH)
    return Q,nA,nS,nM,nD