import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


img = ndimage.imread('/tmp4/hang_data/VOCdevkit/VOC2012/Val_HR_4x/2007_000033.png')
plt.imshow(img, interpolation='nearest')
plt.show()
# Note the 0 sigma for the last axis, we don't wan't to blurr the color planes together!
img_blur = ndimage.gaussian_filter(img, sigma=(2, 2, 2), order=0)
plt.imshow(img_blur, interpolation='nearest')
plt.show()

#img_blur = ndimage.filters.laplace(img)
#plt.imshow(img_blur, interpolation='nearest')
#plt.show()

img_edge = img-img_blur
img_gray = np.average(img_edge, weights=[0.299, 0.587, 0.114], axis=2)
plt.imshow(img_gray, interpolation='nearest', cmap='gray')
plt.show()