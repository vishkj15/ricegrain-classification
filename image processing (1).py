#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import cv2
from PIL import Image
from IPython.display import display,Image
import matplotlib.pyplot as plt


# In[7]:


def get_classification(ratio):
    ratio =round(ratio,1)
    toret=""
    if(ratio>=3 and ratio<3.5):
        toret="Slender"
    elif(ratio>=2.1 and ratio<3):
        toret="Medium"
    elif(ratio>=1.1 and ratio<2.1):
        toret="Bold"
    elif(ratio>0.9 and ratio<=1):
        toret="Round"
    else:
        toret="Dust"
    return toret


# In[8]:


def update_image():
    path="rice.png"
    img1 = cv2.imread(path,0)
    img2=cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#convert into binary
    ret,binary = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
#simple averaging filter
    kernel = np.ones((4,4),np.float32)/9
    dst = cv2.filter2D(binary,-1,kernel)
    
#erosion

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


    erosion = cv2.erode(dst,kernel2,iterations = 1)

#dilation 
    dilation = cv2.dilate(erosion,kernel2,iterations = 1)

#edge detection
    edges = cv2.Canny(dilation,100,200)

### Size detection
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print("No. of rice grains=",len(contours))
    classification1 = {"Slender":0, "Medium":0, "Bold":0, "Round":0, "Dust":0}
    avg1 = {"Slender":0, "Medium":0, "Bold":0, "Round":0, "Dust":0}
    total_ar1=0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        if(aspect_ratio<1):
            aspect_ratio=1/aspect_ratio
#print(round(aspect_ratio,2),get_classification(aspect_ratio))
        classification1[get_classification(aspect_ratio)] += 1
        if get_classification(aspect_ratio) != "Dust":
            total_ar1+=aspect_ratio
        if get_classification(aspect_ratio) != "Dust":
            avg1[get_classification(aspect_ratio)] += aspect_ratio
    avg_ar1=total_ar1/len(contours)
    if classification1['Slender']!=0:
        avg1['Slender'] = avg1['Slender']/classification1['Slender']
    if classification1['Medium']!=0:
        avg1['Medium'] = avg1['Medium']/classification1['Medium']
    if classification1['Bold']!=0:
        avg1['Bold'] = avg1['Bold']/classification1['Bold']
    if classification1['Round']!=0:
        avg1['Round'] = avg1['Round']/classification1['Round']
    cv2.imwrite("img1.jpg", img)
    cv2.imwrite("binary1.jpg", binary)
    cv2.imwrite("dst1.jpg", dst)
    cv2.imwrite("erosion1.jpg", erosion)
    cv2.imwrite("dilation1.jpg", dilation)
    cv2.imwrite("edges1.jpg", edges)
    print("ORIGINAL IMAGE")
    display(Image(filename="rice.png"))
    print("GRAY IMAGE")
    display(Image(filename="img1.jpg"))
    print(" BINARY IMAGE")
    display(Image(filename="binary1.jpg"))
    print("FILTERED IMAGE")
    display(Image(filename="dst1.jpg"))
    print("ERODED IMAGE")
    display(Image(filename="erosion1.jpg"))
    print("DILATION IMAGE")
    display(Image(filename="dilation1.jpg"))
    print("EDGE DETECTION IMAGE")
    display(Image(filename="edges1.jpg"))
    
    return classification1,avg1,avg_ar1

classification,avg,avgr=update_image()


# In[4]:


print("\n")
print("CLASSIFIED CLASS TYPES")
print(classification)
plt.bar(list(classification.keys()),list(classification.values()),color ='blue',width = 0.4)
plt.xlabel("CLASS TYPE")
plt.ylabel("NO of RICE GRAINS")
plt.title("CLASSIFICATION BAR GRAPH")
plt.show()


# In[ ]:




