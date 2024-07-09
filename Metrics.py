import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import hamming_loss
import sys
sys.path.append('/Users/salva/Documents/Salva/UTC/Stage')
from Modelos import *


def priors_label_wise( y):
    distributions = [sum(column) / len(column) for column in y.T]
    return distributions

def f(y_real,y_predict,b=1):
    if np.dot(y_real, y_predict.T)==0  and np.sum(y_real)==0 and np.sum(y_predict)==0 :
        return 1
    numerator =(1+b*b)* np.dot(y_real, y_predict.T)
    denominator = (b*b*np.sum(y_real)) + np.sum(y_predict) 
    return numerator / denominator

def F(y_real,y_predict,b=1):
    suma=0
    for i in range(len(y_real)):
 
       suma+=f(y_real[i],y_predict[i],b)
    return suma/len(y_real)

def return_metrics(Yreal,Ypredictions):
    H=hamming_loss(Yreal,Ypredictions)
    Z=zero_one_loss(Yreal,Ypredictions)
    f1=F(Yreal,Ypredictions)
    M=np.zeros((4,Yreal.shape[1]))
    max_phi=[]
    for i in range(len(Yreal[0])):

        R,m=compute_conditional_risk(Yreal[:,i],Ypredictions[:,i],
                                      2, np.array([[0,1],[1,0]]))
        max_phi.append(np.abs(R[0]-R[1]))
        M[:,i]=np.copy(m.reshape((4,)))
    return H,Z,f1,M,max(max_phi)


def IRLbl(y):
    num_instances,num_labels=y.shape
    sum_list=[sum(y[:,i]) for i in range(num_labels) ]
    max_IRLbl=max(sum_list)
    for i in range(num_labels):
        sum_list[i]=max_IRLbl/sum_list[i]
    return np.array(sum_list)
