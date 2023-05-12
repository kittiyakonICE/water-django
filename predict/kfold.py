import sys
import numpy as np
import pandas as pd
from os import X_OK
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import Counter
from sklearn.model_selection import KFold
from time import sleep
import matplotlib.pyplot as plt

with open('water.csv') as f:
    data = f.read()
    s=str(data,'utf-8')
    data = [[np.asarray(temp[:-3].split(),dtype=np.float64, order='C'),int(temp[-2])] for temp in s.split('\n')[:-1]]
    df = pd.DataFrame(data,columns=['data','class'])
f.close()
x_train = df['data']
y_train = df['class']
with open('water.csv') as f:
    data = f.read()
    s=str(data,'utf-8')
    data = [[np.asarray(i[:-2].split(),dtype=np.float64, order='C'),int(i[-1])] for i in s.split('\n')[:-1]]
    tdf = pd.DataFrame(data,columns=['data','class'])
f.close()
x_test = df['data']
y_test = df['class']
def fn_for_euclidean_distance(train,test):
    
    difference=test-train
    distance=np.sqrt(np.sum(difference**2))    
    return distance
def fn_for_knn(x_test, x_train, y_train,possible_k,sample,method=''):
    
    count = 1
    test_length = len(x_test)

    # Implementing to find best K for KNN algorithm
    if sample=='train': 
        k=temp=1
        y_pred=[[] for _ in range(possible_k)]
    # Implementing to test using best K
    elif sample=='test': 
        k=temp=possible_k
        y_pred=[]
    for test in x_test:
        distances=[]

        for train in x_train:
            distance=fn_for_euclidean_distance(train,test)
            distances.append(distance)

        data={
            'Distances':distances,
            'class':y_train
        }

        df_distance=pd.DataFrame(data)
        df_distance=df_distance.sort_values(by=['Distances'],axis=0)

        k=temp
        
        # For Modified KNN
        if(method=="modified"):
            while(k<=possible_k):
                df_k_distance = np.array(df_distance[:k])
                # # weighted_k =fn_for_finding_weights (k,df_k_distance) 
                # votes = np.zeros(10,dtype=np.float32)

                # for j in range(k):
                #     votes[np.int64(df_k_distance[j][1])]+=weighted_k[j]
                # prediction=np.argmax(votes)

                if sample=='train': 
                    y_pred[k-1].append(prediction)
                elif sample=='test': 
                    y_pred.append(prediction)
    
                k+=1
        
        else:
            while(k<=possible_k):
                df_k_distance=df_distance[:k]
                counter = Counter(y_train[df_k_distance.index])
                prediction=counter.most_common()[0][0]
                
                if sample=='train': 
                    y_pred[k-1].append(prediction)
                elif sample=='test': 
                    y_pred.append(prediction)
                
                k+=1

        count+=1
    
    return y_pred
def fn_for_crossValidation(x_train,y_train,possible_k,method=''):
    y_pred=[]

    num_folds=5
    kf=KFold(n_splits=num_folds,shuffle=True,random_state=42)
    
    accuracies=[0]*(possible_k)
    errors = [0]*(possible_k)
    temp=1
    for train_index, test_index in kf.split(df):

        x_val=x_train[test_index]
        y_val=y_train[test_index]
        x_t=x_train[train_index]
        y_t=y_train[train_index]

        y_pred= fn_for_knn(x_val,x_t,y_t,possible_k,'train',method)
        print("Cross-Validation batch",temp,"completed")

        for k in range(possible_k):
            accuracies[k]+=accuracy_score(y_val, y_pred[k])/num_folds*100
            errors[k]+=mean_squared_error(y_val, y_pred[k])/num_folds*100
        temp+=1
    return accuracies,errors; 

possible_k = 25
print("Cross-Validation STARTED\n")
accuracies,errors = fn_for_crossValidation(x_train,y_train,possible_k)
print("\nCross-Validation COMPLETED\n")
for temp in range(possible_k):
    print('Validation Accuracy for k =',temp+1,"is:",accuracies[temp])

best_k = np.argmax(accuracies)+1
print("Best value for k:",best_k)
print("Maximum Validation Accuracy(Validation Accuracy for Best k):",max(accuracies))
possible_val_k = [i for i in range(1,26)]
plt.plot(possible_val_k, accuracies,color='black', linestyle='dotted', linewidth = 2.5, 
         marker='*', markerfacecolor='green', markersize=9) 

plt.xlabel('K values',labelpad=15)
plt.ylabel('Validation Accuracy',labelpad=20)
plt.title('Validation Accuracy for different Values of K')
plt.show()
print("Best value for k:",best_k)
print("Maximum Validation Accuracy(Validation Accuracy for Best k):",max(accuracies))
y_hat_test=fn_for_knn(x_test,x_train,y_train,best_k,'test')
print("\nTest Accuracy for k as",best_k,"is",accuracy_score(y_test, y_hat_test)*100)