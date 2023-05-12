
from django.urls import path
from .views import *


app_name = "predict"

urlpatterns = [
    path('',home, name='home'),
    path('prediction_page/', predict, name='prediction_page'), #path url index
    path('predict/',predict_chances, name=''), #path url predict.html
    path("edit_predict/<int:id>",predict_edit, name="edit_predict"),
    path("update_predict/<int:id>",update_predict, name=""),
    path('results/', view_results, name='results'), # path url result.html
    path('login/', login_view, name='login-page'),
    path("criteria/",criteria, name='criteria'),
    path('logout/', logout_request,name='logout'),
    path('manage_water_basic/',manage_water_basic_view,name='manage_water'),
    path('add_water_basic/',add_water_basic,name='add_water_basic'),
    path('delete/<int:id>',water_delete,name='water_delete'),
    path('delete_predict/<int:id>',delete_predict,name='delete_predict'),
    path('edit_water_basic/<int:id>',edit_water_basic,name='edit_water_basic'),
    path('update_water_basic/<int:id>',update_water_basic,name=""),
    path('result',result,name='result'),
    path('get_predict/',show_predict_chances,name="submit_prediction"),
    path('test/',test,name="test")
    
  

    
]