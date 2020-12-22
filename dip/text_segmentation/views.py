from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
from .models import *
import os
import cv2
import math
import numpy as np
from PIL import Image as im
from django.core.files.base import ContentFile

#paragraph segmentation
def ParagraphSegmentation(img):
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return image


# Word Segmentation
def WordSegmentation(img):
    image = img.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    #dilation
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    #find contours
    contours, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(image,(x,y),(x+w, y+h),(255,128,0),2)
    return image


#Line Segmentation
def LineSegmentation(img):
	 image = img
	 gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	 
	 ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    
     #dilation
	 kernel = np.ones((5,100), np.uint8)
	 img_dilation = cv2.dilate(thresh, kernel, iterations=1)
	 
     #find contours
	 ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     #sort contours
	 sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	 for i, ctr in enumerate(sorted_ctrs):
		 x, y, w, h = cv2.boundingRect(ctr)
		 roi = image[y:y+h, x:x+w]
		 cv2.rectangle(image,(x,y),(x+w, y+h),(90,0,255),2)
		 
	 return image


#CharSegmentation
def CharSegmentation(img):
    image = img.copy()
    image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #threshold
    res,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) 
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,3))

    #dilation
    dilated = cv2.dilate(thresh,kernel,iterations = 1)
    #find contours
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        cv2.rectangle(image,(x,y),(x+w, y+h),(76,0,153),1)
    
    return image


# Create your views here.
def add_image(request):
    W = Text.objects.all()
    W.delete()
    if request.method == 'POST':
        form = TextForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            val=request.POST.get('All')
            print(val)
            image = Text.objects.last()
            imagefile = image.Img
            image = np.asarray(bytearray(imagefile.read()), dtype="uint8")

            image_encode1 = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_encode2 = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_encode3 = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_encode4 = cv2.imdecode(image, cv2.IMREAD_COLOR)

            img_para = ParagraphSegmentation(image_encode1)
            img_word =  WordSegmentation(image_encode2)
            img_line = LineSegmentation(image_encode3)
            img_char = CharSegmentation(image_encode4)


        cv2.imwrite("text_segmentation/static/text_segmentation/img_para.png",img_para)
        cv2.imwrite("text_segmentation/static/text_segmentation/img_word.png",img_word)
        cv2.imwrite("text_segmentation/static/text_segmentation/img_line.png",img_line)
        cv2.imwrite("text_segmentation/static/text_segmentation/img_char.png",img_char)

        return render(request, 'text_segmentation/img.html', {'form' : form,'success':'yes'})
    else:
        form = TextForm()
        return render(request, 'text_segmentation/img.html',{'form':form})





def disp_word(request):
    return render(request, 'text_segmentation/Disp_word.html')


def disp_para(request):
    return render(request, 'text_segmentation/Disp_para.html')


def disp_line(request):
    return render(request, 'text_segmentation/Disp_line.html')

def disp_char(request):
    return render(request, 'text_segmentation/Disp_char.html')

def disp_all(request):
    return render(request, 'text_segmentation/Disp_all.html')


