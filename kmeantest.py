from sklearn.cluster import KMeans
import numpy as np
import cv2
import matplotlib.pyplot as plt

scale=4
color_num=2

img1 = cv2.imread("test.png",1) #-1RGBA,0Gray 1RGB

w=int(img1.shape[0]/scale)
h=int(img1.shape[1]/scale)

img1 = cv2.resize(img1,(w,h))

#hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#cv2.imshow("test",img)
#cv2.waitKey(3)

img2=img1.reshape(w*h,3)

pred2 = KMeans(n_clusters=color_num).fit_predict(img2)

img3 = pred2.reshape(w,h)

img4 = np.empty((w-1,0))
for x in img3:
    img4=np.append(img4,np.diff(x).reshape(w-1,1),axis=1)

fig = plt.figure(figsize=(10, 30))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.imshow(img1)
ax2.imshow(img3)
ax3.imshow(img4)
plt.show()
