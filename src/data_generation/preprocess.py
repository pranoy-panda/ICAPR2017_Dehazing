import scipy.io as sio
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import add_haze
from helper_functions import patch_generate
from helper_functions import return_A
import os
from sys import path
path.append("../")
from init import * # contains file paths

# loading the folder names
L1=sorted(os.listdir(DEPTH_PATH))
L2=sorted(os.listdir(IMAGE_PATH))

# init constants and variables
stride=15                # stride between patches
k=5                      # patch size
black_px_thrs=  0.75     #threshold for tolerance of percentage of black pixel in the patch
var_red_thrs  = 0.00005
var_green_thrs= 0.00005
var_blue_thrs = 0.00005
count=0
Z1=[]
Z2=[]
Z=[]
T=[]
out=[]

# iterating over variable beta values and folder names
for choice in range(1,5):
    for l1,l2 in zip(L1,L2):
        count+=1
        print('******'+str(count)+'******')
        dep=sio.loadmat(DEPTH_PATH+l1)
        depth=dep[DEPTH_NAME]
        img=misc.imread(IMAGE_PATH+l2)
        if choice==1:
            beta=np.random.uniform(low=0.5,high=1)
        else:
            beta=np.random.uniform()%0.5

        A=return_A()
        t=np.exp(-beta*depth)
        fin=add_haze(img,t,A)
        image,y=patch_generate(A,stride,k,x,depth,beta,black_px_thrs,var_red_thrs,var_green_thrs,var_blue_thrs) # patch_data is a dictionary
        if(image.shape[0]!=0):
            Z.append(image)
            out.append(y)

# converting to numpy arrays and concat. to get desired dim.
Z=np.array(Z)
Z=np.concatenate(Z,axis=0)
out=np.array(out)
out=np.concatenate(out,axis=0)
# saving the generated training data as .npz file
np.savez(NUMPY_DATA_T,X=Z,Y=out)
    






            
