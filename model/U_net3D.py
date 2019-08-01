import numpy as np
from keras.layers import Conv3D, MaxPooling3D, Input, Dropout
from keras.layers import Conv3DTranspose, concatenate, Activation
from keras.models import Model
from keras.layers.core import Lambda
# from keras_contrib.layers import InstanceNormalization
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K



def U_net3D(NUM_CLASS=3):
    ini_filter = 64
    # Build U-Net model
    inputs = Input((IMG_SIZE, IMG_SIZE, col, 3), name='data')
    x = Lambda(lambda x: x, name='input') (inputs)
    box = []
    num_down=4
    for i in range(num_down):
        stage = str(i+1)
        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='1conv_'+stage+'_1') (x)
        x = BatchNormalization(axis=-1, name='1bn_'+stage+'_1')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
        x = BatchNormalization(axis=-1, name='bn_'+stage+'_2')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        box.append(x)   
        ini_filter*=2
        if i<num_down-2: x = MaxPooling3D((2, 2, 2)) (x)
        else: x = MaxPooling3D((2, 2, 1)) (x)

    ini_filter//=2
    box.reverse()

    for l in range(num_down):
        ini_filter//=2
        stage=str(i+l+2)
        if l<num_down-2: x = Conv3DTranspose(ini_filter, (2, 2, 1), strides=(2, 2, 1), padding='same', name='up_'+str(l+1)) (x)
        else: x = Conv3DTranspose(ini_filter, (2, 2, 2), strides=(2, 2, 2), padding='same', name='up_'+str(l+1)) (x)
        x = concatenate([x, box[l]])

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_1') (x)
        x = BatchNormalization(axis=-1, name='bn_'+stage+'_1')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
        x = BatchNormalization(axis=-1, name='bn_'+stage+'_2')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)  
    
    stage = str(int(stage)+1)
    x = Conv3D(64, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
    x = BatchNormalization(axis=-1, name='bn_'+stage+'_2')(x)
    # x = InstanceNormalization()(x)
    x = Activation('relu')(x)  
    x = Conv3D(NUM_CLASS, (1, 1, 1), name='conv3d_1x1') (x)

    if NUM_CLASS==1: outputs = Activation('sigmoid') (x)
    
    model = Model(inputs, x, name='unet3D')
    return model


def U_net3D_weight(inputs, NUM_CLASS=3, bn_train=True, use_bias=True):
    ini_filter = 30
    # Build U-Net model
    x = Lambda(lambda x: x, name='input') (inputs)
    box = []
    num_down=3
    for i in range(num_down):
        stage = str(i+1)
        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_1') (x)
        x = BatchNormalization(axis=-1, name='bn_'+stage+'_1')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
        x = BatchNormalization(axis=-1, name='bn_'+stage+'_2')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        box.append(x)   
        ini_filter*=2
        if i<2: x = MaxPooling3D((2, 2, 2)) (x)
        else: x = MaxPooling3D((2, 2, 1)) (x)

    ini_filter//=2
    box.reverse()

    for l in range(num_down):
        ini_filter//=2
        stage=str(i+l+2)
        if l<num_down-2: x = Conv3DTranspose(ini_filter, (2, 2, 1), strides=(2, 2, 1), padding='same', name='up_'+str(l+1)) (x)
        else: x = Conv3DTranspose(ini_filter, (2, 2, 2), strides=(2, 2, 2), padding='same', name='up_'+str(l+1)) (x)
        x = concatenate([x, box[l]])

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_1') (x)
        x = BatchNormalization(name='bn_'+stage+'_1')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
        x = BatchNormalization(name='bn_'+stage+'_2')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)  
    stage = str(int(stage)+1)
    x = Conv3D(64, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
    x = BatchNormalization(name='bn_'+stage+'_2')(x)
    # x = InstanceNormalization()(x)
    x = Activation('relu')(x)  
    classifier = Conv3D(NUM_CLASS, (1, 1, 1), name='conv3d_1x1') (x)

    return x, classifier

def U_net3D_weight_32col(inputs, NUM_CLASS=3, bn_train=True, use_bias=True):
    ini_filter = 64
    # Build U-Net model
    x = Lambda(lambda x: x, name='input') (inputs)
    box = []
    num_down=4
    for i in range(num_down):
        stage = str(i+1)
        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_1') (x)
        x = BatchNormalization(axis=-1, name='bn_'+stage+'_1')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
        x = BatchNormalization(axis=-1, name='bn_'+stage+'_2')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        box.append(x)   
        ini_filter*=2
        if i<2: x = MaxPooling3D((2, 2, 2)) (x)
        else: x = MaxPooling3D((2, 2, 1)) (x)

    ini_filter//=2
    box.reverse()

    for l in range(num_down):
        ini_filter//=2
        stage=str(i+l+2)
        if l<num_down-2: x = Conv3DTranspose(ini_filter, (2, 2, 1), strides=(2, 2, 1), padding='same', name='up_'+str(l+1)) (x)
        else: x = Conv3DTranspose(ini_filter, (2, 2, 2), strides=(2, 2, 2), padding='same', name='up_'+str(l+1)) (x)
        x = concatenate([x, box[l]])

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_1') (x)
        x = BatchNormalization(name='bn_'+stage+'_1')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
        x = BatchNormalization(name='bn_'+stage+'_2')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)  
    # stage = str(int(stage)+1)
    # x = Conv3D(64, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
    # x = BatchNormalization(name='bn_'+stage+'_2')(x)
    # # x = InstanceNormalization()(x)
    # x = Activation('relu')(x)  
    classifier = Conv3D(NUM_CLASS, (1, 1, 1), name='conv3d_1x1') (x)

    return x, classifier

def U_net3D_weight_32col_v2(inputs, NUM_CLASS=3, bn_train=True, use_bias=True):
    ini_filter = 64
    # Build U-Net model
    x = Lambda(lambda x: x, name='input') (inputs)
    box = []
    num_down=2
    for i in range(num_down):
        stage = str(i+1)
        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_1') (x)
        x = BatchNormalization(axis=-1, name='bn_'+stage+'_1')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
        x = BatchNormalization(axis=-1, name='bn_'+stage+'_2')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        box.append(x)   
        ini_filter*=2
        if i<1: x = MaxPooling3D((2, 2, 2)) (x)
        else: x = MaxPooling3D((2, 2, 1)) (x)

    ini_filter//=2
    box.reverse()

    for l in range(num_down):
        ini_filter//=2
        stage=str(i+l+2)
        if l<num_down-1: x = Conv3DTranspose(ini_filter, (2, 2, 1), strides=(2, 2, 1), padding='same', name='up_'+str(l+1)) (x)
        else: x = Conv3DTranspose(ini_filter, (2, 2, 2), strides=(2, 2, 2), padding='same', name='up_'+str(l+1)) (x)
        x = concatenate([x, box[l]])

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_1') (x)
        x = BatchNormalization(name='bn_'+stage+'_1')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = Conv3D(ini_filter, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
        x = BatchNormalization(name='bn_'+stage+'_2')(x)
        # x = InstanceNormalization()(x)
        x = Activation('relu')(x)  
    # stage = str(int(stage)+1)
    # x = Conv3D(64, (3, 3, 3), padding='same', name='conv_'+stage+'_2') (x)
    # x = BatchNormalization(name='bn_'+stage+'_2')(x)
    # # x = InstanceNormalization()(x)
    # x = Activation('relu')(x)  
    classifier = Conv3D(NUM_CLASS, (1, 1, 1), name='conv3d_1x1') (x)

    return x, classifier