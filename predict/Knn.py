# from ast import operator
from calendar import day_abbr
# from dis import dis
from itertools import count
from operator import imod
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import random
import csv
import operator
import re
from tkinter.tix import COLUMN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from webbrowser import get
import pandas as pd
from socketserver import DatagramRequestHandler
from turtle import color, distance
from urllib import response
from xml.etree.ElementTree import QName



# ดึงข้อมูลจากฐานข้อมูล
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
			for row in dataset:
				i = 1
			for i in range(1,101):
				i += 1
				print("feature--->:",row[:-1],"---->",row[-1])
				break;          
	return dataset

# # ดึงชุดข้อมูลใช้ pd
# dataset = pd.read_csv("water.csv")
# df = pd.read_csv('water.csv')
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# # split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)
# print(dataset)



# # def handleDataset(water,split,trainingSet=[],testSet=[])
# :
# #     with open(water,'r')as csvfile:
# #         lines = reader(csvfile)
# #         dataset = list(lines)
# #         for x in range(len(dataset)-1):
# #             for y in range(4):
# #                 dataset[x][y] = str(dataset[x][y])
# #             if random.random() < split:
# #                 trainingSet.append(dataset[x])
# #             else:
# #                 testSet.append(dataset[x])
# # trainingSet=[]
# # testSet=[]
# # handleDataset(r'water.csv',0.80,trainingSet,testSet)
# # print ('Train: ' + repr(len(trainingSet)))
# # print ('Test: ' + repr(len(testSet)))
            

# # # for row in water:
# # #     i = 1
# #     # for i in range(1,101):
# #     #     i += 1
# #     #     if i <= 101:
# #     #      print(i,"feature--->:",row[:-1],"---->",row[-1])
# #     #     break;
		
# แปลงคอลัมน์สตริงเป็น float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
  
# print(str_column_to_float)

# แปลงคอลัมน์สตริงเป็นจำนวนเต็ม
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
# print(str_column_to_int)

# ค้นหาค่าต่ำสุดและสูงสุดสำหรับแต่ละคอลัมน์
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
# print(dataset_minmax)



# ปรับขนาดคอลัมน์ชุดข้อมูลเป็นช่วง 0-2
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][2] - minmax[i][0])
# print(normalize_dataset)

# แบ่งชุดข้อมูลออกเป็น k เท่า
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
# print(cross_validation_split)

#คำนวณเปอร์เซ็นต์ความแม่นยำ
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
# print(accuracy_metric)

# ประเมินอัลกอริทึมโดยใช้ Spilt
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
# print(evaluate_algorithm)

# คำนวณระยะทางแบบยุคลิดระหว่างเวกเตอร์สองตัว
def euclidean_distance(row1, row2,):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
# print(euclidean_distance)

# หาเพื่อนบ้านที่คล้ายกันมากที่สุด
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
# print(get_neighbors)

# ทายเพื่อนบ้านใกล้เคียง
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
# print(predict_classification)

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)
#  #------------------------------#---------------------------

# # def k_nearest_neighbors(train,test,num_neighbors):
# #     predictions = list()
# #     distance = {}
# #     length = test.shape[1]
# #     for x in range(len(train)):
# #         dist = euclidean_distance(test,train.iloc[x],length)
# #         distance[x] = dist[0]
# #     sortdist = sorted(distance.items(),key=operator.itemgetter(1))
# #     neighbors = []
# #     for x in range(num_neighbors):
# #         neighbors.append(sortdist[x][0])
# #         count = {}
# #         for x in range(len(neighbors)):
# #             response = train.iloc[neighbors[x][-1]]
# #         if response in count:
# #             count[response] +=1
# #         else:
# #             count[response]=1
# #     sortcount = sorted(count.items(),key=operator.itemgetter(1),reversed=True)
# #     return(sortcount[0][0],neighbors)
        
# ทดสอบ knn บนฐานข้อมูล
filename = 'water.csv'
dataset = load_csv(filename)
#ประเมินอัลกอริทึม
n_folds = 3
num_neighbors = 3
row = []
scores = evaluate_algorithm(dataset,n_folds, num_neighbors,)
label = predict_classification(dataset, row, num_neighbors)
print('Distance: %s' % scores)
print('Classification: %s' % (label))
print('Accuracy: %.3f%%' % (sum(scores)/float(len(scores)))) 