#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM
import glob
import skimage.color as color
import matplotlib.pyplot as plt

image = plt.imread("input images/210088.jpg")

#Convert MxNx3 image into Kx3 where K=MxN
img2 = image.reshape((-1,3))

#covariance choices, full, tied, diag, spherical
gmm_model = GMM(n_components=5,covariance_type='full').fit(img2)
segments= gmm_model.predict(img2)
segmented = segments.reshape(image.shape[0], image.shape[1])
plt.imshow(segmented)
plt.show()
img1 = color.label2rgb(segmented, image, kind='avg');


plt.imshow(img1)
plt.show()


# In[6]:


img1.shape


# In[2]:


image = plt.imread("input images/302003.jpg.")
img2 = image.reshape((-1,3))


gmm_model = GMM(n_components=5,covariance_type='full').fit(img2)
segments= gmm_model.predict(img2)
segmented = segments.reshape(image.shape[0], image.shape[1])
plt.imshow(segmented)
plt.show()
img1 = color.label2rgb(segmented, image, kind='avg');


plt.imshow(img1)
plt.show()


# In[3]:


image = plt.imread("input images/im2.jpg")
img2 = image.reshape((-1,3))

gmm_model = GMM(n_components=5,covariance_type='full').fit(img2)
segments= gmm_model.predict(img2)
segmented = segments.reshape(image.shape[0], image.shape[1])
plt.imshow(segmented)
plt.show()
img1 = color.label2rgb(segmented, image, kind='avg');


plt.imshow(img1)
plt.show()


# In[4]:


image = plt.imread("input images/im3.jpg")
img2 = image.reshape((-1,3))

gmm_model = GMM(n_components=5,covariance_type='full').fit(img2)
segments= gmm_model.predict(img2)
segmented = segments.reshape(image.shape[0], image.shape[1])
plt.imshow(segmented)
plt.show()
img1 = color.label2rgb(segmented, image, kind='avg');


plt.imshow(img1)
plt.show()


# In[5]:


image = plt.imread("input images/im4.jpg")
img2 = image.reshape((-1,3))

gmm_model = GMM(n_components=5,covariance_type='full').fit(img2)
segments= gmm_model.predict(img2)
segmented = segments.reshape(image.shape[0], image.shape[1])
plt.imshow(segmented)
plt.show()
img1 = color.label2rgb(segmented, image, kind='avg');


plt.imshow(img1)
plt.show()


# In[ ]:




