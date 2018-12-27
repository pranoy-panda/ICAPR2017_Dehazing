import numpy as np
from scipy import sparse,ndimage
from scipy import misc
import matplotlib.pyplot as plt
from keras.models import load_model
from init import *
from scipy.spatial import distance
from skimage.color import rgb2gray
from skimage import morphology
from skimage import exposure
import sys

### Initialization 
patch_X = 15
patch_Y = 15
stride_X = 10
stride_Y = 10
####

###Loading the model for t and A
model_T = load_model(T_MODEL)
####

### Loading the given image 
def start(im):
  img = misc.imread(im)
  img = img.astype('float32')/255
  (nrow,ncol,_) = img.shape 
  return img,nrow,ncol
####

### Getting the indices for A and t
def index(nrow,ncol):
  x = np.r_[:nrow - patch_X:stride_X]
  y = np.r_[:ncol - patch_Y:stride_Y]
  return x,y
####

### Finding A and t from the model
def get_every(img,x,y,nrow,ncol):
  t = np.zeros((nrow,ncol))
  t = t.astype('float32')
  A = np.zeros((nrow,ncol,3))
  A = A.astype('float32')
  counter_mat = np.zeros((nrow,ncol)) + 1e-4
  Ar=Ag=Ab = 0
  count=0
  counter_mat = counter_mat.astype('float32')
  for x1 in x:
    for y1 in y:
      patch = img[x1:x1+patch_X,y1:y1+patch_Y,:]
      patch = patch[np.newaxis,:,:,:]
      var_ch_t = [np.var(patch[:,:,0]),np.var(patch[:,:,1]),np.var(patch[:,:,2])]
      if np.all(var_ch_t > [0.0005,0.0005,0.0005]):
        out = model_T.predict(patch)
        t[x1:x1+patch_X,y1:y1+patch_Y]+= out[:,0]
        Ar+= out[:,1]
        Ag+= out[:,2]
        Ab+= out[:,3]
        count+= 1
        counter_mat[x1:x1+patch_X,y1:y1+patch_Y]+= 1
  
  t = t/counter_mat
  Ar = Ar/count
  Ag = Ag/count
  Ab = Ab/count
  return t,Ar,Ag,Ab
####

### Ajacency Matrix
def get_laplacian_4neigh(im):
  '''
  im used for the dimension of the Laplacian Matrix and weights of the edges
    for now not taking into account the long range connections
    '''
    # parameter
  min_i_diff_sq = 0.0001
  big_window_frac = 0.15
  big_window_overlap_frac = 0.95
  long_range_i_thr = 0.1
  sampling_skip = 3
  nsample = 5

  [nrow, ncol, nch] = im.shape
  numnode = nrow * ncol
  ind = np.r_[:numnode]
  ind_mat = ind.reshape((nrow, ncol))

  im_r = im.reshape((-1, nch))

    # first compute the adjacency matrix
  adjmat = sparse.csr_matrix((numnode, numnode), dtype='float32')

    # here the arrays are row major
    # right edges
  right_neigh_ind = ind_mat + 1
  right_neigh_excl = right_neigh_ind[:, :-1]
  ind_mat_excl = ind_mat[:, :-1]

    # want || I(x) - I(y) ||^2
  neigh_i_diff = im_r[ind_mat_excl.flatten(), :] \
      - im_r[right_neigh_excl.flatten(), :]

  i_d_norm_sq = np.sum(neigh_i_diff*neigh_i_diff, axis=1)
  right_wt = 1 / np.maximum(i_d_norm_sq, min_i_diff_sq)

  right_edges = sparse.coo_matrix((right_wt, (ind_mat_excl.flatten(),
                                                right_neigh_excl.flatten())),
                                    shape=(numnode, numnode)).tocsr()

  right_edges = right_edges.tocsr()
    # add right and left edges
  adjmat = adjmat + right_edges + right_edges.transpose()

    # down edges
  down_neigh_ind = ind_mat + ncol
  down_neigh_excl = down_neigh_ind[:-1, :]
  ind_mat_excl = ind_mat[:-1, :]

  neigh_i_diff = im_r[ind_mat_excl.flatten(), :] \
      - im_r[down_neigh_excl.flatten(), :]

  i_d_norm_sq = np.sum(neigh_i_diff*neigh_i_diff, axis=1)
  down_wt = 1 / np.maximum(i_d_norm_sq, min_i_diff_sq)

  down_edges = sparse.coo_matrix((down_wt, (ind_mat_excl.flatten(),
                                            down_neigh_excl.flatten())),
                                  shape=(numnode, numnode)).tocsr()

  down_edges = down_edges.tocsr()
    # add down and up edges
  adjmat = adjmat + down_edges + down_edges.transpose()
      # So, adjacency matrix done
  degree = adjmat.sum(axis=1)
  degree_mat = sparse.dia_matrix((degree.flatten(), [0]),
                                   shape=(numnode, numnode))

  laplacian = degree_mat - adjmat

  return laplacian
####

###Getting s and reshaping t_out
def get_s(t_out,nrow,ncol):
  #t_out = (t_out-np.min(t_out))/(np.max(t_out)-np.min(t_out))
  t_out = np.reshape(t_out,(nrow*ncol,1))
  s = np.ones((1,nrow*ncol))
  s = s.astype('float32')
  ind = np.where(t_out==0)[0]
  length = ind.shape[0]
  for i in range(length):
    s[0,ind[i]] = 0
  s = s.tolist()[0]
  s = sparse.diags(s,0,format='csr')
  return s,t_out
####

### Solving the matrix linear equation
def lin_sol(t_out,s,l,nrow,ncol):
  lamda = 0.005
  a = s+(lamda*l)
  b = t_out
  t_final = sparse.linalg.spsolve(a,b)
  t_final = np.reshape(t_final,(nrow,ncol))
  #t_final = (t_final-np.min(t_final))/(np.max(t_final)-np.min(t_final))
  t_final = np.clip(t_final, 0.1, 1)
  return t_final
####

### reconstructig the dehazed image from A,t and original hazy image
def dehaze(t_final,img,Ar,Ag,Ab):
  img_out = np.zeros(img.shape)
  img_out[:,:,0] = Ar + (img[:,:,0] - Ar)/t_final
  img_out[:,:,1] = Ag + (img[:,:,1] - Ag)/t_final
  img_out[:,:,2] = Ab + (img[:,:,2] - Ab)/t_final
  img_out = np.clip(img_out, 0, 1)
  return img_out
####

### Main function
# def main(L):
#   for I in L:
#     img,nrow,ncol = start(TEST_IMAGE_DIR+I)
#     x,y = index(nrow, ncol)                         
#     t,Ar,Ag,Ab = get_every(img, x, y, nrow, ncol)
#     t_out = t
#     l = get_laplacian_4neigh(img)
#     s,t = get_s(t,nrow,ncol)
#     t_final = lin_sol(t, s, l, nrow, ncol)
#     img_out = dehaze(t_final, img, Ar, Ag, Ab)
#     plt.imsave(TEST_OUT_DIR+I, img_out)
#     plt.imsave(TEST_OUT_DIR+'depth'+I,t_final)
def main(im):
    img,nrow,ncol = start(im)
    x,y = index(nrow, ncol)                         
    t,Ar,Ag,Ab = get_every(img, x, y, nrow, ncol)
    t_out = t
    l = get_laplacian_4neigh(img)
    s,t = get_s(t,nrow,ncol)
    t_final = lin_sol(t, s, l, nrow, ncol)
    img_out = dehaze(t_final, img, Ar, Ag, Ab)
    plt.imsave('dehazed_img.jpg', img_out)
    plt.imsave('transmittance_map.jpg',t_final)  
    
if __name__=='__main__':
  #L=os.listdir(TEST_IMAGE_DIR)
  main(sys.argv[1])



