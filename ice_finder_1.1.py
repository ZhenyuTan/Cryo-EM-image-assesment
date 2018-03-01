
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.ndimage.filters import gaussian_filter,minimum_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.filters import sobel
from skimage.transform import resize
from skimage.restoration import denoise_tv_bregman,denoise_bilateral
import mrcfile
import mahotas as mh 
from skimage.io import imread,imshow,show,imsave


def preprocessing_for_icefinder(img_raw):
    #preprocessing the image
    img1=(img_raw-np.min(img_raw))/(np.max(img_raw)-np.min(img_raw))
    img_filtered = gaussian_filter(img1, 40, mode='reflect')
    img_resized=resize(img_filtered,(371,383),order=2,mode='symmetric',preserve_range=True)
    img5=denoise_tv_bregman(img_resized, 0.0001, max_iter=200, eps=0.001, isotropic=True)
    img6=denoise_bilateral(img5,sigma_spatial=25,mode='reflect',multichannel=False)
    img7=(img6-np.min(img6))/(np.max(img6)-np.min(img6))
    return img7




def find_ice(image,sobel_gradient_threshold,size_threshold):
    #sobel_gradient_threshold=0.05
    #size threshold=100
    shape=np.shape(image)
    image1=preprocessing_for_icefinder(image)
    image2=minimum_filter(image1,4, mode='reflect')
    sobel1=sobel(image2)
    image3=sobel1>sobel_gradient_threshold
    image4=binary_fill_holes(mh.morph.dilate(image3))
    #label and clear the image
    labeled, n_nucleus  = mh.label(image4)
    sizes = mh.labeled.labeled_size(labeled) 
    too_small = np.where(sizes < size_threshold)
    labeled = mh.labeled.remove_regions(labeled, too_small)
    relabeled, n_left = mh.labeled.relabel(labeled)
    if n_left >20:
        print("can not locate ice in the image")
    else:
        image5=relabeled>0 #make the binary image
        image6=resize(image5,shape,order=2,mode='symmetric',preserve_range=True)
        return image6


# In[ ]:


img=mrcfile.open('ice11.mrc').data
img2=find_ice(img,0.05,100)
imshow(img2)
show()

