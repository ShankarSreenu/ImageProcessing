from django import forms
from .models import *

class TextForm(forms.ModelForm):
    Img = forms.ImageField()
    class Meta:
        model = Text
        fields = ['Img']
