from .res_net import *
from .U_net3D import *
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, ZeroPadding2D, concatenate, Lambda, ZeroPadding3D, add
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv3D, UpSampling3D, AveragePooling3D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import os
K.set_image_dim_ordering('tf')


def slice(x, h1, h2):
    """ Define a tensor slice function 
    """
    return x[:, :, :, h1:h2,:]
def batch_slice(x, h1):
    """ Define a tensor slice function 
    """
    return x[h1, :, :, :, :]
def slice2d(x, h1, h2):

    tmp = x[h1:h2,:,:,:]
    tmp = tf.transpose(tmp, perm=[1, 2, 0, 3])
    tmp = tf.expand_dims(tmp, 0)
    return tmp
def slice_last(x):

    x = x[:,:,:,:,0]

    return x
def trans(x):

    x = tf.transpose(x, perm=[0,3,1,2,4])

    return x
def trans_back(x):

    x = tf.transpose(x, perm=[0,2,3,1,4])

    return x
def divi_batch(x,h1,h2):
    tmp = x[h1:h2]
    return tmp

def hybrid(args=None):

    #  ************************3d volume input******************************************************************
    img_input = Input(batch_shape=(args.b, args.input_size, args.input_size, args.input_cols, 3), name='volumetric_data')

    #  ************************(batch*d3cols)*2dvolume--2D DenseNet branch**************************************
    input2d_trans = Lambda(trans) (img_input)
    input2d = Lambda(batch_slice, arguments={'h1': 0})(input2d_trans)
    for b in range(1, args.b):
        input2d_tmp = Lambda(batch_slice, arguments={'h1': b})(input2d_trans)
        input2d = concatenate([input2d, input2d_tmp], axis=0)

    #  ******************************stack to 3D volumes *******************************************************
    feature2d, classifer2d = ResNet50_FCN16_weight(input2d, trainable=False, bn_train=False)

    res2d = Lambda(slice2d, arguments={'h1': 0, 'h2':args.input_cols})(classifer2d)
    fea2d = Lambda(slice2d, arguments={'h1':0, 'h2':args.input_cols})(feature2d)
    for b in range(1, args.b):
        res2d_tmp = Lambda(slice2d, arguments={'h1': b*args.input_cols, 'h2':(b+1)*args.input_cols})(classifer2d)
        fea2d_tmp = Lambda(slice2d, arguments={'h1':b*args.input_cols, 'h2':(b+1)*args.input_cols})(feature2d)
        res2d = concatenate([res2d, res2d_tmp], axis=0)
        fea2d = concatenate([fea2d, fea2d_tmp], axis=0)
    #  *************************** 3d DenseNet on 3D volume (concate with feature map )*********************************
    res2d_input = Lambda(lambda x: x * 250)(res2d)
    input3d_ori = Lambda(slice, arguments={'h1': 0, 'h2': args.input_cols})(img_input)
    input3d = concatenate([input3d_ori, res2d_input], axis=4)
    feature3d, classifer3d = U_net3D_weight(input3d)
    print('done2')
    final = add([feature3d, fea2d])

    final_conv = Conv3D(64, (3, 3, 3), padding="same", name='fianl_conv')(final)
    final_conv = Dropout(rate=0.1)(final_conv)
    final_bn = BatchNormalization(name="final_bn")(final_conv)
    final_ac = Activation('relu', name='final_ac')(final_bn)
    classifer = Conv3D(3, (1, 1, 1), padding="same", name='2d3dclassifer')(final_ac)
    classifer = Activation('softmax')(classifer)
    model = Model( inputs = img_input,outputs = classifer, name='auto3d_residual_conv')

    return model