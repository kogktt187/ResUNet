import os
import warnings
from keras import layers
from keras.layers import Input, concatenate
from keras.models import Model
from keras.layers.core import Lambda
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('tf')

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True,pre_str=''):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    pre_str=''
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), trainable=trainable,
                      kernel_initializer='he_normal',
                      name=pre_str+conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same', trainable=trainable,
                      kernel_initializer='he_normal',
                      name=pre_str+conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), trainable=trainable,
                      kernel_initializer='he_normal',
                      name=pre_str+conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               trainable=True,pre_str=''):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal', trainable=trainable,
                      name=pre_str+conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', trainable=trainable,
                      kernel_initializer='he_normal',
                      name=pre_str+conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), trainable=trainable,
                      kernel_initializer='he_normal',
                      name=pre_str+conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal', trainable=trainable,
                             name=pre_str+conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=pre_str+bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet50_FCN16(NUM_CLASS = 3, dim=128, drop=0):
    box=[]
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    img_input = layers.Input((None, None, 3))
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    box.append(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    box.append(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    box.append(x)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    box.append(x)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    box.reverse()
    x = layers.Conv2DTranspose(dim, (16, 16), strides=(16, 16), padding='same', name='fcn16_1') (box[0])
    x = layers.BatchNormalization(name='up_bn_1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(dim//2, (3, 3), padding="same", kernel_initializer="normal", name='up_conv_1')(x)
    if drop>0: x = layers.Dropout(rate=drop)(x)
    x = layers.BatchNormalization(name='up_bn_2')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(NUM_CLASS, (1, 1), name='conv_1x1') (x)
    if NUM_CLASS==1: x = layers.Activation('sigmoid')(x)
    else: x = layers.Activation('softmax')(x)
    # Create model.
    model = Model(img_input, x)
    return model

def ResNet50_FCN16_32col(NUM_CLASS = 3, dim=128, drop=0):
    box=[]
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    img_input = layers.Input((None, None, 3))
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    box.append(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    box.append(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    box.append(x)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    box.append(x)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    box.reverse()
    x = layers.Conv2DTranspose(dim, (16, 16), strides=(16, 16), padding='same', name='fcn16_1') (box[0])
    x = layers.BatchNormalization(name='up_bn_1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(dim//4, (3, 3), padding="same", kernel_initializer="normal", name='up_conv_1')(x)
    if drop>0: x = layers.Dropout(rate=drop)(x)
    x = layers.BatchNormalization(name='up_bn_2')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(NUM_CLASS, (1, 1), name='conv_1x1') (x)
    if NUM_CLASS==1: x = layers.Activation('sigmoid')(x)
    else: x = layers.Activation('softmax')(x)

    # Create model.
    model = Model(img_input, x)
    return model

def ResUNet(NUM_CLASS = 3, up=True, drop=0):
    box=[]
    pre_str = ''
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    img_input = layers.Input((None, None, 3))
    x = layers.ZeroPadding2D(padding=(3, 3), name=pre_str+'conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name=pre_str+'conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=pre_str+'bn_conv1')(x)
    x = layers.Activation('relu')(x)
    box.append(x)
    
    x = layers.ZeroPadding2D(padding=(1, 1), name=pre_str+'pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256],  stage=2, block='c')
    box.append(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    box.append(x)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    box.append(x)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    box.reverse()

    x = layers.UpSampling2D()(x)
    line = layers.Conv2D(2048, (1, 1), padding="same", kernel_initializer="normal", name='line0')(box[0])
    x = layers.add([line, x]) 
    x = layers.Conv2D(512, (3, 3), padding="same", kernel_initializer="normal", name='up_conv1')(x)
    x = layers.BatchNormalization(name='up_bn1')(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D()(x)
    x = layers.add([box[1], x])  
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="normal", name='up_conv2')(x)
    x = layers.BatchNormalization(name='up_bn2')(x)
    x = layers.Activation('relu')(x)
    
    x = layers.UpSampling2D()(x)
    x = layers.add([box[2], x])  
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="normal", name='up_conv3')(x)
    x = layers.BatchNormalization(name='up_bn3')(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D()(x)
    x = layers.add([box[3], x])  
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="normal", name='up_conv4')(x)
    x = layers.BatchNormalization(name='up_bn4')(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="normal", name='up_conv5')(x)
    if drop>0: x = layers.Dropout(rate=drop)(x)
    x = layers.BatchNormalization(name='up_bn5')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(NUM_CLASS, (1, 1), name='conv_1x1') (x)
    if NUM_CLASS==1: x = layers.Activation('sigmoid')(x)
    else: x = layers.Activation('softmax')(x)
    model = Model(img_input, x)
    return model

def identity_block_weight(input_tensor, kernel_size, filters, stage, block, pre_str='', trainable=True, bn_train=False, use_bias=True):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), trainable=trainable,use_bias=use_bias,
                      kernel_initializer='he_normal',
                      name=pre_str+conv_name_base + '2a')(input_tensor)
    if bn_train: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2a')(x)
    else: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2a', trainable=bn_train)(x, training=bn_train)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, trainable=trainable,use_bias=use_bias,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=pre_str+conv_name_base + '2b')(x)
    if bn_train: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2b')(x)
    else: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2b', trainable=bn_train)(x, training=bn_train)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), trainable=trainable,use_bias=use_bias,
                      kernel_initializer='he_normal',
                      name=pre_str+conv_name_base + '2c')(x)
    if bn_train: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2c')(x)
    else: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2c', trainable=bn_train)(x, training=bn_train)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block_weight(input_tensor, 
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               trainable=True, bn_train=False, use_bias=True, pre_str=''):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal', trainable=trainable,use_bias=use_bias,
                      name=pre_str+conv_name_base + '2a')(input_tensor)
    if bn_train: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2a')(x)
    else: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2a', trainable=bn_train)(x, training=bn_train)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', 
                      kernel_initializer='he_normal', trainable=trainable,use_bias=use_bias,
                      name=pre_str+conv_name_base + '2b')(x)
    if bn_train: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2b')(x)
    else: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2b', trainable=bn_train)(x, training=bn_train)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal', trainable=trainable,use_bias=use_bias,
                      name=pre_str+conv_name_base + '2c')(x)
    if bn_train: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2c')(x)
    else: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '2c', trainable=bn_train)(x, training=bn_train)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal', trainable=trainable,use_bias=use_bias,
                             name=pre_str+conv_name_base + '1')(input_tensor)
    if bn_train: shortcut = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '1')(shortcut)
    else: shortcut = layers.BatchNormalization(axis=bn_axis, name=pre_str+bn_name_base + '1', trainable=bn_train)(shortcut, training=bn_train)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResUNet_weight(img_input, NUM_CLASS=3, up=True, drop=0, trainable=False, bn_train=False):
    box=[]
    pre_str=''
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name=pre_str+'conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal', trainable=trainable,
                      name=pre_str+'conv1')(x)
    if bn_train: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+'bn_conv1')(x)
    else: x = layers.BatchNormalization(axis=bn_axis, name=pre_str+'bn_conv1', trainable=bn_train)(x, training=bn_train)
    x = layers.Activation('relu')(x)
    box.append(x)
    
    x = layers.ZeroPadding2D(padding=(1, 1), name=pre_str+'pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block_weight(x, 3, [64, 64, 256],  stage=2, block='a', strides=(1, 1), trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [64, 64, 256],  stage=2, block='b', trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [64, 64, 256],  stage=2, block='c', trainable=trainable, bn_train=bn_train)
    box.append(x)

    x = conv_block_weight(x, 3, [128, 128, 512],  stage=3, block='a', trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [128, 128, 512],  stage=3, block='b', trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [128, 128, 512],  stage=3, block='c', trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [128, 128, 512],  stage=3, block='d', trainable=trainable, bn_train=bn_train)
    box.append(x)

    x = conv_block_weight(x, 3, [256, 256, 1024],  stage=4, block='a', trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [256, 256, 1024],  stage=4, block='b',trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [256, 256, 1024],  stage=4, block='c', trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [256, 256, 1024],  stage=4, block='d', trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [256, 256, 1024],  stage=4, block='e', trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [256, 256, 1024],  stage=4, block='f', trainable=trainable, bn_train=bn_train)
    box.append(x)

    x = conv_block_weight(x, 3, [512, 512, 2048],  stage=5, block='a', trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [512, 512, 2048],  stage=5, block='b', trainable=trainable, bn_train=bn_train)
    x = identity_block_weight(x, 3, [512, 512, 2048],  stage=5, block='c', trainable=trainable, bn_train=bn_train)
    box.reverse()

    x = layers.UpSampling2D()(x)
    line = layers.Conv2D(2048, (1, 1), padding="same", kernel_initializer="normal", name='line0', trainable=trainable)(box[0])
    x = layers.add([line, x]) 
    x = layers.Conv2D(512, (3, 3), padding="same", kernel_initializer="normal", name='up_conv1', trainable=trainable)(x)
    if bn_train: x = layers.BatchNormalization(name='up_bn1')(x)
    else: x = layers.BatchNormalization(name=pre_str+'up_bn1', trainable=bn_train)(x, training=bn_train)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D()(x)
    x = layers.add([box[1], x])  
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="normal", name='up_conv2', trainable=trainable)(x)
    if bn_train: x = layers.BatchNormalization(name='up_bn2')(x)
    else: x = layers.BatchNormalization(name=pre_str+'up_bn2', trainable=bn_train)(x, training=bn_train)
    x = layers.Activation('relu')(x)
    
    x = layers.UpSampling2D()(x)
    x = layers.add([box[2], x])  
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="normal", name='up_conv3', trainable=trainable)(x)
    if bn_train: x = layers.BatchNormalization(name='up_bn3')(x)
    else: x = layers.BatchNormalization(name=pre_str+'up_bn3', trainable=bn_train)(x, training=bn_train)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D()(x)
    x = layers.add([box[3], x])  
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="normal", name='up_conv4', trainable=trainable)(x)
    if bn_train: x = layers.BatchNormalization(name='up_bn4')(x)
    else: x = layers.BatchNormalization(name=pre_str+'up_bn4', trainable=bn_train)(x, training=bn_train)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="normal", name='up_conv5', trainable=trainable)(x)
    if drop>0: x = layers.Dropout(rate=drop)(x)
    if bn_train: x = layers.BatchNormalization(name='up_bn5')(x)
    else: x = layers.BatchNormalization(name=pre_str+'up_bn5', trainable=bn_train)(x, training=bn_train)
    x = layers.Activation('relu')(x)

    classifier = layers.Conv2D(NUM_CLASS, (1, 1), name='conv_1x1', trainable=trainable) (x)
    return x, classifier

def divi_img(x, h1, h2):
    """ Define a tensor slice function 
    """
    return x[:, :, :, h1:h2]

def triple_res(IMG_SIZE=None):
    img_input = layers.Input((IMG_SIZE, IMG_SIZE, 9))
    img_input_1 = Lambda(divi_img, arguments={'h1':0, 'h2':3})(img_input)
    img_input_2 = Lambda(divi_img, arguments={'h1':3, 'h2':6})(img_input)
    img_input_3 = Lambda(divi_img, arguments={'h1':6, 'h2':9})(img_input)
    # img_input_1 = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    # img_input_2 = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    # img_input_3 = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    r1, c = ResNet50_FCN16_weight_32col(img_input_1, NUM=1, trainable=True, bn_train=True)
    r2, c = ResNet50_FCN16_weight_32col(img_input_2, NUM=2, trainable=True, bn_train=True)
    r3, c = ResNet50_FCN16_weight_32col(img_input_3, NUM=3, trainable=True, bn_train=True)
    conbine = layers.concatenate([r1, r2, r3], axis=-1)
    final_conv = layers.Conv2D(64, (3, 3), padding="same", name='fianl_conv')(conbine)
    final_conv = layers.Dropout(rate=0.1)(final_conv)
    final_bn = layers.BatchNormalization(name="final_bn")(final_conv)
    final_ac = layers.Activation('relu', name='final_ac')(final_bn)
    classifer = layers.Conv2D(3, (1, 1), padding="same", name='2d3dclassifer')(final_ac)
    model = Model( inputs = [img_input],outputs = classifer, name='auto3d_residual_conv')
    return model