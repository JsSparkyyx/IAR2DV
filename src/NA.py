import numpy as np
import time

#get decision attribute matrix
def getCharVector(U,labels):
    A = np.zeros((int(labels.max()),U.shape[0]))
    for i in range(int(labels.max())):
        for j in range(U.shape[0]):
            if(int(labels[j]) == i+1):
                A[i,j] = 1
    return A

#get diagnonally relation matrix
def getDiaMatrix(Mu):
    Du = np.zeros((Mu.shape[0],Mu.shape[0]))
    for i in range(Mu.shape[0]):
        Du[i,i] = Mu[i].sum()
    return Du

#get inverse diagnonally relation matrix
def getInverseDiaMatrix(Du):
    Gu = np.zeros((Du.shape[0],Du.shape[0]))
    for i in range((Du.shape[0])):
        Gu[i,i] = 1/Du[i,i]
    return Gu

# 这里AAO的和DAO的不一样
#compute distance matrix and relation matrix
def getMS(U,r):
    M = np.zeros((U.shape[0],U.shape[0]))
    S = np.zeros((U.shape[0],U.shape[0]))
    for i in range(U.shape[0]):
        for j in range(i,U.shape[0]):
            if(i == j):
                M[i,i] = 1
                continue
            d = 0
            for k in range(U.shape[1]):
                d += (U[i,k] - U[j,k])**2
            S[i,j] = d
            S[j,i] = d
            d = d**0.5
            if(d <= r):
                M[i,j] = 1
                M[j,i] = 1
    return M,S

#compute approximation quality
def getQuality(nI):
    POS = np.zeros(nI.shape[0])
    for i in range((nI.shape[0])):
        for j in range((nI.shape[1])):
            if(nI[i,j] >= 0.7):
                POS[i] = 1
    return np.sum(POS)/POS.shape[0]

# run NA
def NA(U,label,r):
    t0 = time.time()
    M,S = getMS(U,r)
    D = getDiaMatrix(M)
    G = getInverseDiaMatrix(D)
    A = getCharVector(U,label)
    I = M.dot(A.T)
    H = np.dot(G,I)
    Q = getQuality(H)
    t1 = time.time()
    return Q,A,S,M,D,t1-t0
