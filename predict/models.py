
from unittest.util import _MAX_LENGTH
from django.db import models
from matplotlib import image



class PredResults(models.Model):
    do = models.FloatField()
    bod = models.FloatField()
    tcb = models.FloatField()
    fcb = models.FloatField()
    nh3n = models.FloatField()
    classification = models.CharField(max_length=40)

    def __str__(self):
        return self.classification
    # แปลงภาษาต่างด้าวเป็นมนุษย์
    
class Water_Basic_Data(models.Model):
    water_source_name = models.TextField()
    province_name = models.TextField()
    area_size = models.TextField()
    water_source_characteristics = models.TextField()
    water_source_information = models.TextField()
    picture_water = models.ImageField(upload_to='images')
    
       
class Water_Image(models.Model):
    picture_id = models.IntegerField()
    picture_water = models.ImageField(upload_to='images')
    waterbasic_id = models.ForeignKey(to='Water_Basic_Data', on_delete=models.CASCADE)
  
class Admin(models.Model):
    User_id = models.TextField(max_length=10)
    FirstName = models.TextField(max_length=255,unique=True)
    LastName = models.TextField(max_length=100,unique=True)
    username = models.TextField(max_length=200,unique=True)
    Password = models.TextField(max_length=100,unique=True)
    

    
    

    


    




