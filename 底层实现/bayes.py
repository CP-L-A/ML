import numpy as np
train_data=[
            [1,'S'],[1,'M'],[1,'M'],[1,'S'],[1,'S'],
            [2,'S'],[2,'M'],[2,'M'],[2,'L'],[2,'L'],
            [3,'L'],[3,'M'],[3,'M'],[3,'L'],[3,'L']
            ]
label=[-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
test_data=[2,'L']
Y_1=0
Y_n1=0
#P={}
def calcaute(datatext,label):
    i=0
    P={}
    global Y_1,Y_n1
    for item in label:
        if item==1:
            Y_1=Y_1+1
        else:
            Y_n1=Y_n1+1
    P_1=Y_1/len(label)
    P_0=1-P_1
    for item in datatext:
        if item[0]==1:
            if label[i]==1:
                P['1_1']=(P.get('1_1',0)+1)/Y_1
            elif label[1]==-1:
                P['1_-1']=(P.get('1_-1',0)+1)/Y_n1
        elif item[0]==2:
            if label[i]==1:
                P['2_1']=(P.get('2_1',0)+1)/Y_1
            elif label[1]==-1:
                P['2_-1']=(P.get('2_-1',0)+1)/Y_n1        
        else:
            if label[i]==1:
                P['3_1']=(P.get('3_1',0)+1)/Y_1
            elif label[1]==-1:
                P['3_-1']=(P.get('3_-1',0)+1)/Y_n1  
        if item[1]=='S':
            if label[i]==1:
                P['S_1']=(P.get('S_1',0)+1)/Y_1
            elif label[1]==-1:
                P['S_-1']=(P.get('S_-1',0)+1)/Y_n1
        if item[1]=='M':
            if label[i]==1:
                P['M_1']=(P.get('M_1',0)+1)/Y_1
            elif label[1]==-1:
                P['M_-1']=(P.get('M_-1',0)+1)/Y_n1
        if item[1]=='L':
            if label[i]==1:
                P['L_1']=(P.get('L_1',0)+1)/Y_1
            elif label[1]==-1:
                P['L_-1']=(P.get('L_-1',0)+1)/Y_n1
        i=i+1
    return P,P_1,P_0
def classify(dataset,P,P_1,P_0):
    if dataset[0]==1:
        p1_1=np.log(P['1_1'])
        p1_n1=np.log(P['1_-1'])
    elif dataset[0]==2:
        p1_1=np.log(P['2_-1'])
        p1_n1=np.log(P['2_-1'])
    else:
        p1_1=np.log(P['3_1'])
        p1_n1=np.log(P['3_-1'])
    if dataset[1]=='S':
        p2_1=np.log(P['S_1'])
        p2_n1=np.log(P['S_-1'])
    elif dataset[1]=='M':
        p2_1=np.log(P['M_1'])
        p2_n1=np.log(P['M_-1'])
    else:
        p2_1=np.log(P['L_1'])
        p2_n1=np.log(P['L_-1'])
    p1=p1_1+p2_1+np.log(P_1)
    pn1=p1_n1+p2_n1+np.log(P_0)
    if p1>pn1:
        print(str(dataset)+"类别为1")
    else:
        print(str(dataset)+"类别为-1")
def main():
    [P,P_1,P_n1]=calcaute(train_data,label)
    classify(test_data,P,P_1,P_n1)
main()