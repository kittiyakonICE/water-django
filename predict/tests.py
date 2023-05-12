import pandas as pd
import numpy as np
import math

# get data from dataset
df = pd.read_csv("water.csv")
df = df.sample(frac = 1)

x = np.array(df.drop(["class"],1))
y = np.array(df["class"])

# data = preprocessing.normalize(df,axis=0)
# scaled_df = pd.DataFrame(data,names)
# scaled_df.head()

n_train = math.floor(0.8 * x.shape[0])
n_test = math.ceil((1-0.8) * x.shape[0])

fold = 5


for i in range(fold):
        x_train = x[:n_train]
        x_test = x[:n_test]
        y_train = y[:n_train]
        y_test = y[:n_test]
        
        print("Fold: ",i+1 ,"Accuracy : ")
        
        
        
# def normalize_dataset(dataset,minmax):
#         for row in dataset:
#                 for i in range(len(row)):
#                         row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# for i in range(0,5):
        # df = pd.read_csv("water.csv")
        # df = df.sample(frac = 1)

        # X = np.array(df.drop(["class"],1))
        # y = np.array(df["class"])

        # n_train = math.floor(0.8 * X.shape[0])
        # n_test = math.ceil((1-0.8) * X.shape[0])

        # X_train = X[:n_train]
        # y_train = y[:n_train]
        # X_test = X[n_train:]
        # y_test = y[n_train:]
        
        # clf = clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test) #ทดสอบโมเดล
        # y_score = clf.score(X, y) 
        # acc = accuracy_score(y_test, y_pred)*100
        # result.append(acc)
        # row.append(i+1)
        # print("Fold:",i+1," accuracy",acc,"%")
        
# perm = sum(result)/5
# dict = {'Fold': row, 'Accuracy': result, 'Performance': perm}
# df1 = pd.DataFrame(dict)
# # df1.to_csv('performance.csv')

# print("Performance:" ,perm, "%")
