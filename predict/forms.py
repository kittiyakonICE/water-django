from django import forms

from .models import PredResults, Water_Basic_Data

class PickForm(forms.ModelForm):

    class Meta:
        model = PredResults
        fields = ('do', 'bod','tcb','fcb','nh3n')

# class UserImage(forms.ModelForm):  
#     class meta:  
#         # To specify the model to be used to create form  
#         models = Water_Basic_Data  
#         # It includes all the fields of model  
#         fields = '__all__'  
        
class Update_Water_Basic(forms.ModelForm):
    
    class Meta:
        model = Water_Basic_Data
        fields = ('water_source_name','province_name','area_size','water_source_characteristics','water_source_information')