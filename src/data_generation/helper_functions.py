import scipy.io as sio
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

'''
This function generates image patches and the corresponding transmittance map of the patch.
Input: stride, patch_size(k), beta, color thresholds(var_red_thrs,var_green_thrs,var_blue_thrs), image(x)
Output: image patches(var)-> [num_samples,patch_size,patch_size,3], transmittance for each patch -> [num_samples,patch_size,patch_size]
'''
def patch_generate(stride,k,x,depth,beta,black_px_thrs,var_red_thrs,var_green_thrs,var_blue_thrs):
    [rows,cols,_]=x.shape
    #print(rows,cols)
    #if k>=rows or k>=cols:
    #    return 0

    # init variables
    row_begin=0
    row_end=k
    col_begin=0
    col_end=k
    counter=0
    var=[]
    var_hist=[]
    pat_data={}
    t=[]
    num_patches=(rows*cols)/(k**2) # finding the number of patches 
    img=np.ones((14400,k,k,3))     # limiting the size of the variable (for preventing mem. issues)
    
    # patch generation loop
    while (row_end<rows+1 and ~(cols-col_end<k-1)):
        roi=x[row_begin:row_end,col_begin:col_end,:]   # region of interest
        doi=depth[row_begin:row_end,col_begin:col_end] # depth of interest
        useful_px=doi>0
        doi=doi.flatten()
        var=np.array([np.var(roi[:,:,0]),np.var(roi[:,:,1]),np.var(roi[:,:,2])])
        # discarding the patches with less variance i.e. smooth patches, or patches where depth info is not available
        if roi.shape==(k,k,3) and np.sum(useful_px)>k*k*black_px_thrs and np.all(var>[var_red_thrs,var_green_thrs,var_blue_thrs]):
            img[counter,:,:,:]= roi
            #print(np.sum(useful_px))
            t.append(np.median(doi))
            counter+=1
        if cols-col_end<k-1 and (cols-col_begin-stride)<k:
            col_begin = 0
            col_end   = k
            row_begin = row_begin + stride
            row_end   = row_end + stride
        else:
            col_begin = col_begin+stride
            col_end   = col_end+stride
            
    t=np.array(t) # converting the list to an array
    t=np.exp(-beta*t) # generating transmittance from depth 
    [num_patch]=t.shape
    var=img[0:num_patch,:,:,:].copy()
    return var,t

'''
This function generates hazzy image by using the "Haze Model Equation"
Input: image(img),transmittance map(t), Environmental Illumination(A)
Output: hazzy image (fin)
'''
def add_haze(img,t,A):
    img.astype(float)
    img=img/255
    
    Ar=A[0]
    Ag=A[1]
    Ab=A[2]
    
    im_r=img[:,:,0]
    im_g=img[:,:,1]
    im_b=img[:,:,2]
    
    # using the haze model equation
    fin_r=im_r*t + Ar*(1-t)
    fin_g=im_g*t + Ag*(1-t)
    fin_b=im_b*t + Ab*(1-t)

    fin=np.ones(img.shape)
    fin[:,:,0]=fin_r
    fin[:,:,1]=fin_g
    fin[:,:,2]=fin_b

    return fin

'''
returns uniformly random sampled value of A [range is from 0 to 1]
Input: None
Output: Environmental Illumination(A)-> [[Ar,Ag,Ab]]
'''
def return_A():
    
    R=np.random.uniform(0.4,1)
    G=np.random.uniform(0.4,1)
    B=np.random.uniform(0.4,1)
    
    A=np.array([[R,G,B]])
    return A    