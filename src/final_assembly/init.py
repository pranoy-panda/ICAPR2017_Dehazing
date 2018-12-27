import keras.backend as K
import os

###Initialzation
PATCH_X_A=100
PATCH_Y_A=100
PATCH_Z_A=10
NUM_CHANNEL=3
####

###Loading Model
T_MODEL="D:/ISI/Dehaze/hope15.h5"
####

###Loading the data for training
NUMPY_DATA_T="D:/ISI/Dehaze/new_data_t2.npz"
####

###Output directory path
TEST_IMAGE_DIR="D:/ISI/Dehaze/test_images/input/new/"
TEST_OUT_DIR="D:/ISI/Dehaze/test_images/input/"
####
