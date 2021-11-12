from AR import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import cross_val_score
from sklearn import preprocessing
import pandas as pd
import argparse

# Run AAO,NA in experiment setting
# nU: objects with all attributes, label: decision attributes, r: radius
def runExpAAO(nU,label,r):
    attr = []
    obj = []
    time_IA = []
    time_NA = []
    for i in range(5,11):
        attr.append(round(nU.shape[1]*i*0.1))
        obj.append(round(nU.shape[0]*i*0.1))
    attr_p = attr[0]
    obj_p = obj[0]
    naU = nU[:obj_p,:attr_p]
    nalabel = label[:obj_p]
    Q,A,S,M,D,time_na = NA(naU,nalabel,r)
    for i in range(1,len(attr)):
        attr_p = attr[i-1]
        attr_l = attr[i]
        obj_p = obj[i-1]
        obj_l = obj[i]
        U = nU[:obj_p,:attr_p]
        iU = nU[:obj_l,:attr_l]
        dU = iU[obj_p:,:]
        dC = iU[:,attr_p:]
        labelp = label[:obj_p]
        labels = label[:obj_l]
        t0 = time.time()
        Q_1,A,S,M,D = AAO(U,dU,iU,dC,labels,A.T,S,M,D,r)
        A = A.T
        t1 = time.time()
        time_IA.append(t1-t0)
        naU = nU[:obj_l,:attr_l]
        nalabel = label[:obj_l]
        Q_2,a,b,c,d,time_na = NA(naU,nalabel,r)
        time_NA.append(time_na)
        print("Round {}, NA Q: {:.3f}, IA Q: {:.3f}".format(str(i),Q_1,Q_2))
        print("Round {}, NA takes {:.3f}s, IA takes {:.3f}s".format(str(i),time_NA[-1],time_IA[-1]))
    return time_IA,time_NA

# Run DAO,NA in experiment setting
def runExpDAO(nU,label,r):
    attr = []
    obj = []
    time_IA = []
    time_NA = []
    for i in range(10,4,-1):
        attr.append(round(nU.shape[1]*i*0.1))
        obj.append(round(nU.shape[0]*i*0.1))
    attr_p = attr[0]
    obj_p = obj[0]
    naU = nU[:obj_p,:attr_p]
    nalabel = label[:obj_p]
    Q,A,S,M,D,time_na = NA(naU,nalabel,r)
    for i in range(len(attr)-1):
        attr_p = attr[i]
        attr_l = attr[i+1]
        obj_p = obj[i]
        obj_l = obj[i+1]
        U = nU[:obj_p,:attr_p]
        dC = nU[:,attr_l:attr_p]
        label1 = label[:obj_l]
        r1 = [i for i in range(obj_l,obj_p)]
        r1.insert(0,-1)
        dU = np.array(r1)
        t0 = time.time()
        Q_1,A,S,M,D = DAO(U,dU,dC,label1,A.T,S,M,D,r)
        t1 = time.time()
        time_IA.append(t1-t0)
        naU = nU[:obj_l,:attr_l]
        nalabel = label[:obj_l]
        Q_2,A,S,M,D,time_na = NA(naU,nalabel,r)
        time_NA.append(time_na)
        print("Round {}, NA Q {:.3f}, IA Q {:.3f}".format(str(i),Q_1,Q_2))
        print("Round {}, NA takes {:.3f}s, IA takes {:.3f}s".format(str(i+1),time_NA[-1],time_IA[-1]))
    return time_IA,time_NA

# Run IARAAO,NIAR in experiment setting
def runExpIARAAO(nU,label,r):
    attr = []
    obj = []
    time_IAR = []
    red_IAR = []
    time_NIAR = []
    red_NIAR = []
    for i in range(5,11):
        attr.append(round(nU.shape[1]*i*0.1))
        obj.append(round(nU.shape[0]*i*0.1))
    attr_p = attr[0]
    obj_p = obj[0]
    naU = nU[:obj_p,:attr_p]
    nalabel = label[:obj_p]
    red, time = NIAR(naU,nalabel,r)
    for i in range(1,len(attr)):
        attr_p = attr[i-1]
        attr_l = attr[i]
        obj_p = obj[i-1]
        obj_l = obj[i]
        iU = nU[:obj_l,:attr_l]
        U = iU[:obj_p,red]
        dU = iU[obj_p:,:]
        dC = iU[:,attr_p:]
        labelp = label[:obj_p]
        labels = label[:obj_l]
        red, time = IARAAO(U,dU,iU,dC,labels,labelp,attr_p,attr_l,red,r)
        time_IAR.append(time)
        red_IAR.append(copy.deepcopy(red))
        naU = nU[:obj_l,:attr_l]
        nalabel = label[:obj_l]
        red_p, time_p = NIAR(naU,nalabel,r)
        time_NIAR.append(time_p)
        red_NIAR.append(red_p)
    return red_IAR,red_NIAR,time_IAR,time_NIAR

# Run IARDAO,NIAR in experiment setting
def runExpIARDAO(nU,label,r):
    attr = []
    obj = []
    time_IAR = []
    red_IAR = []
    time_NIAR = []
    red_NIAR = []
    for i in range(10,4,-1):
        attr.append(round(nU.shape[1]*i*0.1))
        obj.append(round(nU.shape[0]*i*0.1))
    attr_p = attr[0]
    obj_p = obj[0]
    naU = nU[:obj_p,:attr_p]
    nalabel = label[:obj_p]
    red, time = NIAR(naU,nalabel,r)
    for i in range(len(attr)-1):
        attr_p = attr[i]
        attr_l = attr[i+1]
        obj_p = obj[i]
        obj_l = obj[i+1]
        iU = nU[:obj_p,:attr_p]
        U = iU[:obj_p,red]
        r1 = [i for i in range(obj_l,obj_p)]
        r1.insert(0,-1)
        dU = np.array(r1)
        setb = list(set([i for i in range(attr_l,attr_p)])&set(red))
        dC = iU[:,setb]
        labelp = label[:obj_p]
        labels = label[:obj_l]
        red, time = IARDAO(U,dU,iU,dC,labels,labelp,obj_l,attr_p,attr_l,red,r)
        time_IAR.append(time)
        red_IAR.append(copy.deepcopy(red))
        naU = nU[:obj_l,:attr_l]
        nalabel = label[:obj_l]
        red_p, time_p = NIAR(naU,nalabel,r)
        time_NIAR.append(time_p)
        red_NIAR.append(red_p)
    return red_IAR,red_NIAR,time_IAR,time_NIAR

# Run IARAADO,NIAR in experiment setting
def runExpIARAADO(nU,label,r):
    attr = []
    obj = []
    time_IAR = []
    red_IAR = []
    time_NIAR = []
    red_NIAR = []
    for i in range(10,4,-1):
        obj.append(round(nU.shape[0]*i*0.1))
    for i in range(5,11):
        attr.append(round(nU.shape[1]*i*0.1))
    attr_p = attr[0]
    obj_p = obj[0]
    naU = nU[:obj_p,:attr_p]
    nalabel = label[:obj_p]
    red, time = NIAR(naU,nalabel,r)
    for i in range(len(attr)-1):
        attr_p = attr[i]
        attr_l = attr[i+1]
        obj_p = obj[i]
        obj_l = obj[i+1]
        iU = nU[:obj_p,:attr_l]
        U = iU[:obj_p,red]
        r1 = [i for i in range(obj_l,obj_p)]
        r1.insert(0,-1)
        dU = np.array(r1)
        dC = iU[:,attr_p:]
        labelp = label[:obj_p]
        labels = label[:obj_l]
        red, time = IARAADO(U,dU,iU,dC,labels,labelp,obj_l,attr_p,attr_l,red,r)
        time_IAR.append(time)
        red_IAR.append(copy.deepcopy(red))
        naU = nU[:obj_l,:attr_l]
        nalabel = label[:obj_l]
        red_p, time_p = NIAR(naU,nalabel,r)
        time_NIAR.append(time_p)
        red_NIAR.append(red_p)
    return red_IAR,red_NIAR,time_IAR,time_NIAR

# Run IARDAAO,NIAR in experiment setting
def runExpIARDAAO(nU,label,r):
    attr = []
    obj = []
    time_IAR = []
    red_IAR = []
    time_NIAR = []
    red_NIAR = []
    for i in range(10,4,-1):
        attr.append(round(nU.shape[1]*i*0.1))
    for i in range(5,11):
        obj.append(round(nU.shape[0]*i*0.1))
    attr_p = attr[0]
    obj_p = obj[0]
    naU = nU[:obj_p,:attr_p]
    nalabel = label[:obj_p]
    red, time = NIAR(naU,nalabel,r)
    for i in range(1,len(attr)):
        attr_p = attr[i-1]
        attr_l = attr[i]
        obj_p = obj[i-1]
        obj_l = obj[i]
        iU = nU[:obj_l,:attr_p]
        U = iU[:obj_p,red]
        dU = iU[obj_p:,:]
        setb = list(set([i for i in range(attr_l,attr_p)])&set(red))
        dC = iU[:,setb]
        labelp = label[:obj_p]
        labels = label[:obj_l]
        red, time = IARDAAO(U,dU,iU,dC,labels,labelp,attr_p,attr_l,red,r)
        time_IAR.append(time)
        red_IAR.append(copy.deepcopy(red))
        naU = nU[:obj_l,:attr_l]
        nalabel = label[:obj_l]
        red_p, time_p = NIAR(naU,nalabel,r)
        time_NIAR.append(time_p)
        red_NIAR.append(red_p)
    return red_IAR,red_NIAR,time_IAR,time_NIAR

# Evaluate results
# nU: objects with all attributes, label: decision attributes, red_IAR: reduct of IAR, red_NIAR: reduct of NIAR
def KNNRun(nU,label,red_IAR,red_NIAR):
    KNN_IAR = KNeighborsClassifier()
    KNN_NIAR = KNeighborsClassifier()
    obj = []
    accuracy_IAR = []
    accuracy_NIAR = []
    for i in range(6,11):
        obj.append(round(nU.shape[0]*i*0.1))
    for i in range(len(red_IAR)):
        KNN_IAR = KNeighborsClassifier()
        KNN_NIAR = KNeighborsClassifier()
        obj_l = obj[i]
        X_IAR = nU[:obj_l,red_IAR[i]]
        X_NIAR = nU[:obj_l,red_NIAR[i]]
        labels = label[:obj_l]
        scores_IAR = cross_val_score(KNN_IAR,X_IAR,labels,cv=5,scoring='accuracy')
        scores_nIAR = cross_val_score(KNN_NIAR,X_NIAR,labels,cv=5,scoring='accuracy')
        accuracy_IAR.append(scores_IAR.mean())
        accuracy_NIAR.append(scores_nIAR.mean())
    print(accuracy_IAR)
    print(accuracy_NIAR)
    return accuracy_IAR,accuracy_NIAR

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='./data/SPF.csv',
                        help='Input data path')
    
    parser.add_argument('--exp', type=str, default='IARAAO',
                        help='Input exp type')
    parser.add_argument('--r', type=float, default=1,
                        help='Input radius')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # object format <attribute 1>,...,<attribute n>,<label>
    # data = pd.read_csv(args.data,delimiter='\t',header=None)
    data = pd.read_csv(args.data,delimiter='\t',header=None).sample(700)
    data = np.array(data)
    np.random.shuffle(data)

    # normalize data attributes
    min_max_scaler = preprocessing.MinMaxScaler() 
    label = data[:,-1]
    nU = data[:,:-1]
    nU = min_max_scaler.fit_transform(data[:,:-1])

    # label index starts from 1
    if(min(label) == 0):
        label[label == 0] = max(label) + 1

    if args.exp == "AAO":
        time_IA,time_NA = runExpAAO(nU,label,args.r)
    elif args.exp == "DAO":
        time_IA,time_NA = runExpDAO(nU,label,args.r)
    elif args.exp == "IARAAO":
        red_IAR,red_NIAR,time_IAR,time_NIAR = runExpIARAAO(nU,label,args.r)
        accuracy_IAR,accuracy_NIAR = KNNRun(nU,label,red_IAR,red_NIAR)
    elif args.exp == "IARDAO":
        red_IAR,red_NIAR,time_IAR,time_NIAR = runExpIARDAO(nU,label,args.r)
        accuracy_IAR,accuracy_NIAR = KNNRun(nU,label,red_IAR,red_NIAR)
    elif args.exp == "IARAADO":
        red_IAR,red_NIAR,time_IAR,time_NIAR = runExpIARAADO(nU,label,args.r)
        accuracy_IAR,accuracy_NIAR = KNNRun(nU,label,red_IAR,red_NIAR)
    elif args.exp == "IARDAAO":
        red_IAR,red_NIAR,time_IAR,time_NIAR = runExpIARDAAO(nU,label,args.r)
        accuracy_IAR,accuracy_NIAR = KNNRun(nU,label,red_IAR,red_NIAR)
    else:
        print("Experiment should be AAO, DAO, IARAAO, IARDAO, IARAADO or IARDAAO")

