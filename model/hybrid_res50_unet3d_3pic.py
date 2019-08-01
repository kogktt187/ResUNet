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



def tumor_slice(x):
    return x[:,:,:,:,2:3]

def slice(x, h1, h2):
    """ Define a tensor slice function 
    """
    return x[:, :, :, h1:h2,:]
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

def hybrid_3pic(args, drop=0.2):

    #  ************************3d volume input******************************************************************
    img_input = Input(batch_shape=(args.b, args.input_size, args.input_size, args.input_cols, 1), name='volumetric_data')

    #  ************************(batch*d3cols)*2dvolume--2D DenseNet branch**************************************
    for b in range(args.b):
        small_input =  Lambda(divi_batch, arguments={'h1': b, 'h2': b+1})(img_input)
        input2d = Lambda(slice, arguments={'h1': 0, 'h2': 2})(small_input)
        single = Lambda(slice, arguments={'h1':0, 'h2':1})(small_input)
        input2d = concatenate([single, input2d], axis=3)
        for i in range(args.input_cols - 2):
            input2d_tmp = Lambda(slice, arguments={'h1': i, 'h2': i + 3})(small_input)
            input2d = concatenate([input2d, input2d_tmp], axis=0)
            if i == args.input_cols - 3:
                final1 = Lambda(slice, arguments={'h1': args.input_cols-2, 'h2': args.input_cols})(small_input)
                final2 = Lambda(slice, arguments={'h1': args.input_cols-1, 'h2': args.input_cols})(small_input)
                final = concatenate([final1, final2], axis=3)
                input2d = concatenate([input2d, final], axis=0)
        input2d = Lambda(slice_last)(input2d)
        if b==0: a = input2d
        else: a = concatenate([a, input2d], axis=0)
    input2d = a
    #  ******************************stack to 3D volumes *******************************************************
    feature2d, classifer2d = ResUNet_weight(input2d, trainable= False, bn_train=False)
#     classifer2d = Activation('softmax')(classifer2d)
    for b in range(args.b):
        small_class2d = Lambda(divi_batch, arguments={'h1': b*args.input_cols, 'h2': (b+1)*args.input_cols})(classifer2d)
        small_feature2d = Lambda(divi_batch, arguments={'h1': b*args.input_cols, 'h2': (b+1)*args.input_cols})(feature2d)
        res2d = Lambda(slice2d, arguments={'h1': 0, 'h2': 1})(small_class2d)
        fea2d = Lambda(slice2d, arguments={'h1':0, 'h2':1})(small_feature2d)
        for j in range(args.input_cols - 1):
            score = Lambda(slice2d, arguments={'h1': j + 1, 'h2': j + 2})(small_class2d)
            fea2d_slice = Lambda(slice2d, arguments={'h1': j + 1, 'h2': j + 2})(small_feature2d)
            res2d = concatenate([res2d, score], axis=3)
            fea2d = concatenate([fea2d, fea2d_slice], axis=3)
        if b==0: 
            r = res2d
            f = fea2d
        else: 
            r = concatenate([r, res2d], axis=0)      
            f = concatenate([f, fea2d], axis=0)   
    res2d = r
    fea2d = f
    #  *************************** 3d DenseNet on 3D volume (concate with feature map )*********************************
    res2d_input = Lambda(lambda x: x*250 )(res2d)
    input3d_ori = Lambda(slice, arguments={'h1': 0, 'h2': args.input_cols})(img_input)
#     res2d_input = Lambda(tumor_class)(res2d_input)
    input3d = concatenate([input3d_ori, res2d_input], axis=4)
    
    # feature3d, classifer3d = U_net3D_weight_32col(input3d)

    input3d_1 = Conv3D(32, (3, 3, 3), padding="same")(input3d)
    input3d_1 = BatchNormalization(axis=-1)(input3d_1)
    input3d_1 = Activation('relu')(input3d_1)
    input3d = concatenate([input3d, input3d_1], axis=4)
    input3d = Conv3D(32, (3, 3, 3), padding="same")(input3d)
    input3d = BatchNormalization(axis=-1)(input3d)
    feature3d = Activation('relu')(input3d)
    
    final = add([feature3d, fea2d])
#     final = concatenate([feature3d, fea2d], axis=4)
    final_conv = Conv3D(64, (3, 3, 3), padding="same", name='fianl_conv')(final)
    if drop>0: final_conv = Dropout(rate=drop)(final_conv)
    final_bn = BatchNormalization(name="final_bn")(final_conv)
    final_ac = Activation('relu', name='final_ac')(final_bn)
    classifer = Conv3D(3, (1, 1, 1), padding="same", name='2d3dclassifer')(final_ac)
    classifer = Activation('softmax')(classifer)

    model = Model( inputs = img_input,outputs = classifer, name='auto3d_residual_conv')

    return model