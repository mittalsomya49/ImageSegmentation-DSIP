import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
#Clustering is the task of dividing the population (data points) 
# into a number of groups, such that data points in the same groups are more similar to other
# data points in that same group than those in other groups. These groups are known as clusters.
#One of the most commonly used clustering algorithms is k-means. 
# Here,the k represents the number of clusters.

One of the most commonly used clustering algorithms is k-means. Here, the k represents the number of clusters.

def kmeans_cl(image, k=5):
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	pixel_values = image.reshape((-1, 3))
	pixel_values = np.float32(pixel_values)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
	t0=time.time()
	_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	t1=time.time()
	centers = np.uint8(centers)
	segmented_image = centers[labels.flatten()]
	segmented_image = segmented_image.reshape(image.shape)
	#segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
	plt.imshow(segmented_image)
	plt.show()
	#plt.imsave('kmeans_color.png', segmented_image)
	print('Time taken by kmeans to segment image of size {} is: {}'.format(image.shape, t1-t0))

image = cv2.imread('input_images/im2.jpg')
kmeans_cl(image, k=256)
