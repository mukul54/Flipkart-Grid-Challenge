
# coding: utf-8

# In[ ]:

import pandas as pd
import cv2
from random import shuffle
import numpy as np
from keras import losses
from keras import layers
from keras import models
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Add
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import time
import tensorflow as tf
from keras import optimizers
from keras.initializers import glorot_uniform
import pickle

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# %matplotlib inline

img_height = 480
img_width = 640
img_channels = 4

#
# network params
#

cardinality = 32


def residual_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    # conv1
    x = layers.Conv2D(32, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 64, 128, _project_shortcut=project_shortcut)

    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 128, 256, _strides=strides)

    # conv4
    for i in range(23):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides)

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 1024, _strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4)(x)

    return x


image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
network_output = residual_network(image_tensor)


# define the checkpoint
""" filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint] """

#model1 = model(input_shape = (224, 224, 4), classes = 4)
model1 = models.Model(inputs=[image_tensor], outputs=[network_output])

#model1 = load_model("model.h5")

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
def customLoss(y_True,y_Pred):
    return (100*losses.mean_squared_error(y_True, y_Pred))

model1.compile(optimizer = adam,loss= ['mean_squared_error'], metrics=['mae'])

pklfile= 'modelweights.pkl'
try:
	f= open(pklfile) 	#Python 2
                   	 
	weigh= pickle.load(f)
	f.close()
except:

	f= open(pklfile, 'rb') 	#Python 3            	 
	weigh= pickle.load(f)           	 
	f.close()

#use set_weights to load the modelweights into the model architecture
model1.set_weights(weigh)


for x in range(48*15):
    cnt = ((x%48)*500)+500
    X_train = np.load('data_imgtrain'+str(cnt)+'.npy')
    Y_train = np.load('data_lab'+str(cnt)+'.npy')
    """  for ori in range(1,4):
        X_data = np.load('data_imgtrain'+str(cnt+(ori*24000))+'.npy')
        Y_data = np.load('data_lab'+str(cnt+(ori*24000))+'.npy')
        X_train = np.concatenate((X_train,X_data), axis = 0)
        Y_train = np.concatenate((Y_train,Y_data), axis = 0)
    np.random.seed(42)
    np.random.shuffle(X_train)
    np.random.seed(42)
    np.random.shuffle(Y_train) """
    model1.fit(X_train, Y_train, epochs=1, batch_size=2)
    print('*********')
    print((x+1)/48)
    if ((x+1)%48) == 0:
        weigh= model1.get_weights()
        pklfile= 'modelweights'+str((x+1)/48)+'.pkl'
        try:
	        fpkl= open(pklfile, 'wb')	#Python 3	 
	        pickle.dump(weigh, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
	        fpkl.close()
        except:
    	    fpkl= open(pklfile, 'w')	#Python 2 	 
    	    pickle.dump(weigh, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
    	    fpkl.close()

""" df = pd.read_csv('test.csv')
c1 = 0
d1 = 0
for index, row in df.iterrows():
    path = './images/' + row['image_name']
    img = cv2.imread(path)
    edges = cv2.Canny(img,50,255)
    #print(edges[:,:,np.newaxis].shape)
    img = np.concatenate((img,edges[:,:,np.newaxis]), axis = 2)
    img = img[np.newaxis,:,:,:]
    prediction = model1.predict(img)
    if index == 0:
        predictions = prediction
    else:
        predictions = np.concatenate((predictions,prediction), axis = 0)
    print(index)
np.save('preds.npy', predictions) """

#now, use pickle to save your model weights, instead of .h5
#for heavy model architectures, .h5 file is unsupported.

