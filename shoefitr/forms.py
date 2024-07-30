from django import forms
from .models import Shoes, data

class shoesform(forms.ModelForm):
    name = forms.CharField(max_length=100,required=False)
    user_id = forms.CharField(max_length=50)
    shop_id = forms.CharField(max_length=50)
    shoes_model_id = forms.CharField(max_length=50)

    class Meta:
        model = Shoes
        fields = '__all__'


class uploadshoedataForm(forms.ModelForm):
    file = forms.FileField(label="Upload New Data File")

    class Meta:
        model = data
        fields = {'file'}
