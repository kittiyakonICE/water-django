from asyncore import read
from csv import reader
from dataclasses import replace
from fileinput import filename
import json
from predict.knnModel import load_csv, predict_classification, str_column_to_float

from django.contrib.auth.forms import AuthenticationForm #add this

from django.shortcuts import render

import pandas as pd
from predict.forms import PickForm, Update_Water_Basic

from .models import PredResults, Water_Basic_Data
from django.contrib.auth import authenticate
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout #add this
from django.core.paginator import Paginator
from math import sqrt            
import numpy as np
import pandas as pd
from django.http import JsonResponse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




# method เพื่อให้ predict.html เป็นหน้าแรกของเว็บ
def predict(request):
    return render(request, 'predict.html')

def home(request):
    if 'q' in request.GET:
        q=request.GET['q']
        posts=Water_Basic_Data.objects.filter(water_source_name=q)
    else:
        posts=Water_Basic_Data.objects.all()
    # Pagintion
    paginator=Paginator(posts,5)
    page_number=request.GET.get('page')
    posts_obj=paginator.get_page(page_number)
    return render(request,'home.html',{'posts':posts_obj})


def delete_predict(request,id):
    delete = PredResults.objects.get(id=id)
    delete.delete()
    result = PredResults.objects.all()
    return render(request,'results.html',{"data":result})

        

def predict_user(request):
    return render(request, 'predict_user.html')


def update_predict(request,id):
    update = PredResults.objects.get(id=id)
    form= PickForm(request.POST,instance=update)
    if form.is_valid():
        form.save()
        messages.success(request,"Updated Successfully...!")
        return render(request,"edit_predict.html",{"edit_predicts":update}) 


def predict_edit(request,id):
    getstudentdeails=PredResults.objects.get(id=id)
    return render(request,'edit_predict.html',{"edit_predict":getstudentdeails})




#method predict_chances เป็น method ที่รับ จากฟอร์มใน result.html

def predict_chances(request):
    if request.method == "POST":
        if request.POST.get('do') and request.POST.get('bod') and request.POST.get('tcb') and request.POST.get('fcb') and request.POST.get('nh3n'):
            data = PredResults()
        
            data.do = request.POST.get('do')
            data.bod = request.POST.get('bod')
            data.tcb = request.POST.get('tcb')
            data.fcb = request.POST.get('fcb')
            data.nh3n = request.POST.get('nh3n')
            data.save()
   
    return render(request,'predict.html')


#method view_results method ที่ส่ง dataset แสดงผลการจำแนก render ไปยัง result.html
def view_results(request):
    # ส่งคำจำแนกและแสดงทั้งหมดไปยัง result.html ใน method view_result 
    contact_list = PredResults.objects.all()
    paginator = Paginator(contact_list, 20) # Show 25 contacts per page.

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'results.html', {'page_obj': page_obj})
       

def surface(request):
    return render(request, 'surface.html')

  
def criteria(request):
    return render(request, 'criteria.html')


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return render(request,'home.html')
            else:
                messages.error(request,"ชื่อผู้ใช้หรือรหัสผ่านผิด")
        else:
             messages.error(request,"ชื่อผู้ใช้หรือรหัสผ่านผิด")
    form = AuthenticationForm()
    return render(request, 'login.html', context={"login_form":form})



def logout_request(request):
    logout(request)
    messages.info(request, "You have successfully logged out.") 
    return redirect("predict:home")


def manage_water_basic_view(request):
    data = Water_Basic_Data.objects.all()
   
    return render(request,'water_basic_view.html', {"data":data})

def test(request):
    f = open('Accuracy.json')
    f1 = open('FoldAcc.json')
    data = json.load(f)
    data1 = json.load(f1)
    
    acc = str(round(data['Accuracy'],2))
    fold = list()
    for i in data1:
        for key,value in i.items():
            fold.append(str(round(value,2)))
    return render(request, 'test.html', { "data" : acc, "fold": fold })

def update_water_basic(request,id):
    update = Water_Basic_Data.objects.get(id=id)
    form = Update_Water_Basic(request.POST,instance=update)
    data = Water_Basic_Data()
    
   
    if form.is_valid():
        form.save()
        return render(request,'edit_water_basic.html',{'edit_water_basic':update})



def edit_water_basic(request,id):
    edit_water_basic = Water_Basic_Data.objects.get(id=id)
    return render(request,'edit_water_basic.html',{"edit_water_basic":edit_water_basic})

    
def add_water_basic(request):
    if request.method == "POST":
        if request.POST.get('water_source_name') and request.POST.get('province_name') and request.POST.get('area_size') and request.POST.get('water_source_characteristics') and request.POST.get('water_source_information'):
            data = Water_Basic_Data()

            
            if len(request.FILES) != 0:
                data.picture_water = request.FILES['image']
                data.save()
                  
            data.water_source_name = request.POST.get('water_source_name')
            data.province_name = request.POST.get('province_name')
            data.area_size = request.POST.get('area_size')
            data.water_source_characteristics = request.POST.get('water_source_characteristics')
            data.water_source_information = request.POST.get('water_source_information')
            data.save()
            return redirect("predict:add_water_basic")
    else:
        return render(request,'add_water_basic.html')
            

def water_delete(request,id):
    delete = Water_Basic_Data.objects.get(id=id)
    delete.delete()
    result = Water_Basic_Data.objects.all()
    
    return render(request,"water_basic_view.html",{"data":result})


        
        
#--------------------------------user---------------------------#
def result(request):
    return render(request, 'result.html')


datasets = pd.read_csv("water.csv")

features = datasets.drop(columns='class',axis = 1)
labels= datasets['class']


x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2) #,stratify=count_class,random_state=2)
# print(x_test)

scale =StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)
def most_common(lst):
    return max(set(lst), key=lst.count)

def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))

class KNeighborsClassifier:
    def __init__(self, k=3, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print("ป้อนชุดข้อมูล New")
        
    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
            print('class:' + repr(neighbors))
            # print(distances)
        return list(map(most_common, neighbors))
    
    
    # def classify(self,neighbors):
    #     neighbors = []
    #     for i in distances:
    #         neighbors.append(self.y_train[i[0]])
    #     print(neighbors[0]) #ตัด1[0]
        
        
#object of knn
knn =  KNeighborsClassifier(k=1) 
knn.fit(x_train,y_train)
# knn.predict(np.array([6.85,0.68,3400,1100]).reshape(1,4))
# print(x_train[0])

def predict_new(request):
    if request.method == "POST":
        if request.POST.get('do') and request.POST.get('bod') and request.POST.get('tcb') and request.POST.get('fcb') and request.POST.get('nh3n'):
            data = PredResults()
        
            data.do = request.POST.get('do')
            data.bod = request.POST.get('bod')
            data.tcb = request.POST.get('tcb')
            data.fcb = request.POST.get('fcb')
            data.nh3n = request.POST.get('nh3n')
   
        return render(request,'predict.html')
    
# x_new = scale.transform(x_new)
# result = knn.predict(x_new)
#     if result == 0:
#          print("")
#     else:
#         print("")
#     predict_new()

###################################################################



#------------------------------------------------------------------------------------------------------------

def show_predict_chances(request):
    filename = 'watersTest.csv'
    k = 3
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
        
    if request.POST.get('action') == 'post':
        test_data = list()
        # Receive data from client
        do = float(request.POST.get('do'))
        bod = float(request.POST.get('bod'))
        tcb = float(request.POST.get('tcb'))
        fcb = float(request.POST.get('fcb'))
        nh3n = float(request.POST.get('nh3n'))
        
        test_data = [do, bod, tcb, fcb, nh3n]
        
        prediction = predict_classification(dataset, test_data, k)
        # Unpickle model
        # model = pd.read_pickle(r"D:\โปรเจค Ai\วิจัย KNN\demo\prediction\KNNModel.pkl")
        # # Make prediction
        # result = model.predict([[do, bod, tcb, fcb, nh3n]])

        print(prediction)
        
        if prediction == 'Q2':
            prediction = 'Q2 ระดับคุณภาพน้ำดีมาก'
        elif prediction == 'Q3':
            prediction = 'Q3 ระดับคุณภาพน้ำดี'
        elif prediction == 'Q4':
            prediction = 'Q4 ระดับคุณภาพน้ำพอใช้'
        else:
             prediction = 'Q5 ระดับคุณภาพน้ำเสื่อม'
        
        

        # classification = result[0]

        # PredResults.objects.create(do=do, bod=bod, tcb=tcb,
        #                            fcb=fcb,nh3n=nh3n, classification=classification)
        return JsonResponse({'result': prediction, 'do': do,
                             'bod': bod, 'tcb': tcb, 'fcb': fcb, 'nh3n': nh3n},
                            safe=False)

