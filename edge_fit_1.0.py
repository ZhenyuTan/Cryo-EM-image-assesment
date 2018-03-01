
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.ndimage.filters import gaussian_filter,median_filter
from skimage.filters import sobel
from skimage.transform import resize
from skimage.restoration import denoise_tv_bregman
import mrcfile
import mahotas as mh 
from skimage.io import imread,imshow,show
from scipy import optimize


# In[ ]:


def preprocessing(img_raw):
    #preprocessing the image
    img1=(img_raw-np.min(img_raw))/(np.max(img_raw)-np.min(img_raw))
    img_filtered = gaussian_filter(img1, 30, mode='reflect')
    img_resized=resize(img_filtered,(371,383),order=2,mode='symmetric',preserve_range=True)
    img5=denoise_tv_bregman(img_resized, 0.0001, max_iter=200, eps=0.001, isotropic=True)
    img6=median_filter(img5, size=50, mode='reflect')
    return img6

                         


# In[ ]:


def find_edge(img_processed):
    #use sobel filter to find edge
    sobel1= sobel(img_processed)
    sobel_norm=(sobel1-np.min(sobel1))/(np.max(sobel1)-np.min(sobel1))
    img10=sobel_norm>((np.max(sobel_norm)+np.mean(sobel_norm))/3) 
    img11=mh.morph.dilate(img10)
    return img11
        


# In[ ]:


def check_img_status(image_edge):
    #after finding edge check the image to see if it is ok to fit the circle
    labeled, n_nucleus  = mh.label(image_edge)
    if n_nucleus>10:
        return 0
    else:
        return 1 
    


# In[ ]:


def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate,args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sqrt(np.sum((Ri - R)**2))
    return xc, yc, R, residu


# In[ ]:


def fit_edge(image_aftercheck):
    #use circle to fit the edge
    ys,xs = np.where(image_aftercheck==1)
    xc_fit,yc_fit,R_fit,error= leastsq_circle(xs,ys)
    return xc_fit,yc_fit,R_fit,error


# In[ ]:


def check_circle_fit(radius_fit,fit_error):
    #check the circle_fit result
    #if the error is too large and the radius is too small, then do not use the fit result
    if fit_error<120 and radius_fit>450:
        return 1
    else:
        return 0 


# In[ ]:


def draw_circle(a,b,c):
    #draw the circle on a new image used for filter particles 
    circle=np.zeros((371,383))
    for i in range(383):
        for j in range(371):
            distance=(i-a)**2+(j-b)**2
            if distance<=(c-10)**2:
                circle[j][i]=1
    return circle
    


# In[ ]:



def edge_find_and_fit(img):
    #main program 
    img_1=preprocessing(img)
    img_2=find_edge(img_1)
    if check_img_status(img_2)==1:
        x_fit,y_fit,radius,error=fit_edge(img_2)
        if check_circle_fit(radius,error)==0:
            print('not a good circle fit')
        else:
            img_3=draw_circle(x_fit,y_fit,radius)
            img_4=resize(img_3,(3710,3838),order=2,mode='symmetric',preserve_range=True)
            #right now i think we should save the img_4 as the same name as the micrograph 
            #if we want to use it to filter particles 
            imshow(img_4)
            show()
            
        
    else:
        print('not a good image to fit')
        
        

    





# In[ ]:


img=mrcfile.open('holes9.mrc').data
edge_find_and_fit(img)


# In[ ]:



                                


# In[ ]:



        

