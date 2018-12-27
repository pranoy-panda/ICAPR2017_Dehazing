import numpy as np
import keras
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten , Input
from keras.layers import Conv2D,MaxPooling2D
from keras.models import Model
from keras import optimizers , metrics
from keras.layers.core import Activation
from init import *
from keras.layers.merge import Concatenate
from keras.layers.convolutional import  UpSampling2D
from keras import regularizers
from keras import initializers

batch_size = 1000
num_classes = 100
epochs=5
img_rows, img_cols = 15, 15

input_shape = (img_rows, img_cols, 3)

x=Input(shape=(input_shape))
b=keras.initializers.Constant(value=0.01)

#Start of the network
#1st layer
x1=Conv2D(8,(1,1),activation="relu",bias_initializer=b)(x)
x1=Conv2D(8,(7,7),activation="relu",bias_initializer=b)(x1)
x1=Conv2D(16,(5,5),activation="relu",bias_initializer=b)(x1)
x1=Conv2D(8,(3,3),activation="relu",bias_initializer=b)(x1)

#2nd layer
x2=Conv2D(8,(1,1),activation="relu",bias_initializer=b)(x)
x2=Conv2D(8,(7,7),activation="relu",bias_initializer=b)(x2)
x2=Conv2D(16,(5,5),activation="relu",bias_initializer=b)(x2)

#3rd layer
x3=Conv2D(8,(1,1),activation="relu",bias_initializer=b)(x)
x3=Conv2D(8,(5,5),activation="relu" ,bias_initializer=b)(x3)
x3=Conv2D(8,(3,3),activation="relu" ,bias_initializer=b)(x3)
x3=Conv2D(8,(3,3),activation="relu" ,bias_initializer=b)(x3)
x3=Conv2D(8,(3,3),activation="relu" ,bias_initializer=b)(x3)

#1st Concatenate
x3=Concatenate(axis=-1)([x2,x3])
x3=Conv2D(8,(3,3),activation="relu" ,bias_initializer=b)(x3)

#2nd Concatenate
x4=Concatenate(axis=-1)([x1,x3])

#Flattening and Dense Layer
x4=Flatten()(x4)
x4=Dropout(rate=0.3)(x4)
x4=Dense(40,bias_initializer=b)(x4)
x4=Activation("relu")(x4)
x4=Dense(4)(x4)
x4=Activation("relu")(x4)

##End of Network

#Load data
D=np.load(NUMPY_DATA_T)
X=D['X']
Y=D['Y']
##

#Training info
model=Model(inputs=x,outputs=x4)
model.compile(loss='mse',optimizer='Adadelta',metrics=[metrics.mae])
model.fit(X,Y,batch_size=batch_size,epochs=epochs,verbose=1,shuffle='batch')
model.save(T_MODEL)
####
######Total number of parameters---> 25036
