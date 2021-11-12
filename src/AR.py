from AAO import *
from DAO import *
from DAAO import *
from AADO import *
import copy

def getArg(U,dU,nU,label,C,B,A,S,M,D,r):
    CmB = list(set(C)-set(B))
    Qm = -1
    mA,mS,mM,mD = A,S,M,D
    index = 0
    for i in CmB:
        Q,nA,nS,nM,nD = AAO(U,dU,U,nU[:,i],label,A,S,M,D,r)
        #print(i,Q)
        if(Q > Qm):
            Qm = Q
            index = i
            mA,mS,mM,mD = nA,nS,nM,nD
    return index,Qm,mA,mS,mM,mD

'''
    Run algorithm NIAR
    naU: objects
    nalabel: labels
    r: radius
'''
def NIAR(naU,nalabel,r):
    C_dC = [i for i in range(naU.shape[1])]
    ept = np.array([])
    t0 = time.time()
    Q_U_dU_C_dC,nA,nS,nM,nD,_ = NA(naU,nalabel,r)
    nA = nA.T
    print(Q_U_dU_C_dC)
    B_prime = []
    for i in range(len(C_dC)):
        Q_U_dU_B_dC = DAAO(naU,ept,naU,naU[:,i],nalabel,nA,nS,nM,nD,r)[0]
        if(Q_U_dU_B_dC != Q_U_dU_C_dC):
            print("select attribute "+str(i))
            B_prime.append(i)
    red_prime = B_prime
    Q_U_dU_red,nA,nS,nM,nD,_ = NA(naU[:,red_prime],nalabel,r)
    nA = nA.T
    while(Q_U_dU_red < Q_U_dU_C_dC):
        c,Q_U_dU_red,nA,nS,nM,nD = getArg(naU[:,red_prime],ept,naU,nalabel,C_dC,red_prime,nA,nS,nM,nD,r)
        red_prime.append(c)
        print("append attribute "+str(c))
    t1 = time.time()
    print("Algorithm NIAR took {:.2f}s".format(t1 - t0))
    red_prime.sort()
    print(red_prime)
    return red_prime,t1-t0

'''
    Run algorithm IARAAO
    U: objects of last round
    dU: new objects
    iU: all objects
    dC: new attributes
    labels: all labels
    labelp: labels of last round
    attr_p: new attributes' indexes start from
    attr_l: new attributes' indexes end at
    red: reduct of last round
    r: radius
'''
def IARAAO(U,dU,iU,dC,labels,labelp,attr_p,attr_l,red,r):
    M,S = getMS(U,r)
    D = getDiaMatrix(M)
    A = getCharVector(U,labelp)
    A = A.T
    t0 = time.time()
    B = red
    for i in range(attr_p,attr_l):
        B.append(i)
    C_dC = [i for i in range(attr_l)]
    ept = np.array([])
    Q_U_dU_B_dC,nA,nS,nM,nD = AAO(U,dU,iU[:,B],dC,labels,A,S,M,D,r)
    Q_U_dU_C_dC = NA(iU,labels,r)[0]
    print(Q_U_dU_B_dC,Q_U_dU_C_dC)
    while(Q_U_dU_B_dC < Q_U_dU_C_dC):
        c,Q_U_dU_B_dC,nA,nS,nM,nD = getArg(iU[:,B],ept,iU,labels,C_dC,B,nA,nS,nM,nD,r)
        B.append(c)
        print("append attribute "+str(c))
    red_prime = B
    for i in red_prime:
        Q_U_dU_red_dC,mA,mS,mM,mD = DAAO(iU,ept,iU,iU[:,i],labels,nA,nS,nM,nD,r)
        if(Q_U_dU_red_dC >= Q_U_dU_B_dC):
            print("delete attribute "+str(i))
            nA,nS,nM,nD = mA,mS,mM,mD
            print(i,red_prime)
            red_prime.remove(i)
    t1 = time.time()
    print("Algorithm IAR took {:.2f}s".format(t1 - t0))
    red_prime.sort()
    print(red_prime)
    return red_prime,t1-t0

'''
    Run algorithm IARAAO
    U: objects of last round with reduct 
    dU: indexes of deleted objects
    iU: objects of last round with all attributes U = iU[:,red]
    dC: new attributes
    labels: all labels
    labelp: labels of last round
    obj_l: deleted objects' indexes end at
    attr_p: new attributes' indexes start from
    attr_l: new attributes' indexes end at
    red: reduct of last round
    r: radius
'''
def IARAADO(U,dU,iU,dC,labels,labelp,obj_l,attr_p,attr_l,red,r):
    M,S = getMS(U,r)
    D = getDiaMatrix(M)
    A = getCharVector(U,labelp)
    A = A.T
    t0 = time.time()
    B = red
    for i in range(attr_p,attr_l):
        B.append(i)
    C_dC = [i for i in range(attr_l)]
    ept = np.array([])
    Q_U_dU_B_dC,nA,nS,nM,nD = AADO(iU,dU,dC,labels,A,S,M,D,r)
    Q_U_dU_C_dC = NA(iU[:obj_l,:],labels,r)[0]
    print(Q_U_dU_B_dC,Q_U_dU_C_dC)
    while(Q_U_dU_B_dC < Q_U_dU_C_dC):
        c,Q_U_dU_B_dC,nA,nS,nM,nD = getArg(iU[:obj_l,B],ept,iU[:obj_l,:],labels,C_dC,B,nA,nS,nM,nD,r)
        B.append(c)
        print("append attribute "+str(c))
    red_prime = B
    for i in red_prime:
        Q_U_dU_red_dC,mA,mS,mM,mD = DAAO(iU[:obj_l,:],ept,iU[:obj_l,:],iU[:,i],labels,nA,nS,nM,nD,r)
        if(Q_U_dU_red_dC >= Q_U_dU_B_dC):
            print("delete attribute "+str(i))
            nA,nS,nM,nD = mA,mS,mM,mD
            #print(i,red_prime)
            red_prime.remove(i)
    t1 = time.time()
    print("Algorithm IAR took {:.2f}s".format(t1 - t0))
    red_prime.sort()
    print(red_prime)
    return red_prime,t1-t0

'''
    Run algorithm IARDAO
    U: objects of last round with reduct 
    dU: indexes of deleted objects
    iU: objects of last round with all attributes U = iU[:,red]
    dC: deleted attributes
    labels: all labels
    labelp: labels of last round
    obj_l: deleted objects' indexes end at
    attr_p: deleted attributes' indexes start from
    attr_l: deleted attributes' indexes end at
    red: reduct of last round
    r: radius
'''
def IARDAO(U,dU,iU,dC,labels,labelp,obj_l,attr_p,attr_l,red,r):
    M,S = getMS(U,r)
    D = getDiaMatrix(M)
    A = getCharVector(U,labelp)
    A = A.T
    t0 = time.time()
    B = list(set(red)-set([i for i in range(attr_l,attr_p)]))
    C_dC = [i for i in range(attr_l)]
    ept = np.array([])
    Q_U_dU_B_dC,nA,nS,nM,nD = DAO(iU,dU,dC,labels,A,S,M,D,r)
    Q_U_dU_C_dC = NA(iU[:obj_l,:attr_l],labels,r)[0]
    print(Q_U_dU_B_dC,Q_U_dU_C_dC)
    while(Q_U_dU_B_dC < Q_U_dU_C_dC):
        c,Q_U_dU_B_dC,nA,nS,nM,nD = getArg(iU[:obj_l,B],ept,iU[:obj_l,:attr_l],labels,C_dC,B,nA,nS,nM,nD,r)
        B.append(c)
        print("append attribute "+str(c))
    red_prime = B
    for i in red_prime:
        Q_U_dU_red_dC,mA,mS,mM,mD = DAAO(iU[:obj_l,:attr_l],ept,iU[:obj_l,:attr_l],iU[:obj_l,i],labels,nA,nS,nM,nD,r)
        if(Q_U_dU_red_dC >= Q_U_dU_B_dC):
            print("delete attribute "+str(i))
            nA,nS,nM,nD = mA,mS,mM,mD
            print(i,red_prime)
            red_prime.remove(i)
    t1 = time.time()
    print("Algorithm IAR took {:.2f}s".format(t1 - t0))
    red_prime.sort()
    print(red_prime)
    return red_prime,t1-t0

'''
    Run algorithm IARDAAO
    U: objects of last round
    dU: new objects
    iU: all objects
    dC: deleted attributes
    labels: all labels
    labelp: labels of last round
    attr_p: deleted attributes' indexes start from
    attr_l: deleted attributes' indexes end at
    red: reduct of last round
    r: radius
'''
def IARDAAO(U,dU,iU,dC,labels,labelp,attr_p,attr_l,red,r):
    M,S = getMS(U,r)
    D = getDiaMatrix(M)
    A = getCharVector(U,labelp)
    A = A.T
    t0 = time.time()
    B = list(set(red)-set([i for i in range(attr_l,attr_p)]))
    C_dC = [i for i in range(attr_l)]
    ept = np.array([])
    Q_U_dU_B_dC,nA,nS,nM,nD = DAAO(U,dU,iU[:,B],dC,labels,A,S,M,D,r)
    Q_U_dU_C_dC = NA(iU[:,:attr_l],labels,r)[0]
    print(Q_U_dU_B_dC,Q_U_dU_C_dC)
    while(Q_U_dU_B_dC < Q_U_dU_C_dC):
        c,Q_U_dU_B_dC,nA,nS,nM,nD = getArg(iU[:,B],ept,iU,labels,C_dC,B,nA,nS,nM,nD,r)
        B.append(c)
        print("append attribute "+str(c))
        print(Q_U_dU_B_dC,Q_U_dU_C_dC)
    red_prime = B
    for i in red_prime:
        Q_U_dU_red_dC,mA,mS,mM,mD = DAAO(iU,ept,iU,iU[:,i],labels,nA,nS,nM,nD,r)
        if(Q_U_dU_red_dC >= Q_U_dU_B_dC):
            print("delete attribute "+str(i))
            nA,nS,nM,nD = mA,mS,mM,mD
            # print(i,red_prime)
            red_prime.remove(i)
    t1 = time.time()
    print("Algorithm IAR took {:.2f}s".format(t1 - t0))
    red_prime.sort()
    print(red_prime)
    return red_prime,t1-t0