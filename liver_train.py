import os
root_folder = os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from keras.optimizers import SGD
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model

from model.res_net import *
from loss.liver_loss import *
from liver_gene import *

import argparse
parser = argparse.ArgumentParser(description='Liver Training')
#  other paras
parser.add_argument('-b', type=int, default=40)
parser.add_argument('-input_size', type=int, default=256)
parser.add_argument('-thread_num', type=int, default=14)
args = parser.parse_args()

learning_rate = 1e-3
momentum = 0.9
BATCH_SIZE = args.b
EPOCH = 600
IMG_SIZE = args.input_size
NUM_CLASS = 1
lower_HU, upper_HU = -200, 250
pre_tarin = True
drop = 0.3
thread_num = args.thread_num
data_path = 'data/'

def train_and_predict():
    model_weight = ResUNet(NUM_CLASS=NUM_CLASS, drop=drop)

    if pre_tarin:
        load_model_name = 'weight/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        model_weight.load_weights(load_model_name, by_name=True)
        # model_weight =  multi_gpu_model(model_weight, gpus=2)
    else:
        load_model_name='Experiments/liver_model/weights.XX-0.XXX.h5'
        # model_weight = multi_gpu_model(model_weight, gpus=2)
        model_weight.load_weights(load_model_name)



    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=momentum, nesterov=True)
    model_weight.compile(optimizer=sgd, loss=[F_score_loss(NUM_CLASS=NUM_CLASS)], 
                         metrics=[Dice_1(NUM_CLASS=NUM_CLASS), recall_1(NUM_CLASS=NUM_CLASS)])

    train_people = list(range(131)) 
    img_path = os.path.join(data_path, 'image_mat')
    liver_location_dir = os.path.join(data_path, 'liver_slice/')
    data_info=[]
    for i in train_people:
        f = open(liver_location_dir+str(i)+'.txt')
        liver_location = f.read()
        liver_location = liver_location.split('\n')[:-1]

        person_path = os.path.join(img_path, str(i))
        final_slice = len(os.listdir(person_path))-1
        data_info.append([i,final_slice,liver_location])
        f.close()
    print(len(data_info))
    model_checkpoint = ModelCheckpoint('Experiments/liver_model/weights.{epoch:02d}-{Dice:.3f}.h5', monitor='loss', verbose=1,
                                       save_best_only=True,save_weights_only=True,mode='min', period=1)

    model_weight.fit_generator(generator = liver_generator(data_info, IMG_SIZE=IMG_SIZE, thread_num= thread_num, data_path=data_path,
                                                           lower_HU=lower_HU, upper_HU=upper_HU, batch_size=BATCH_SIZE),
                               steps_per_epoch=1000, epochs=EPOCH, verbose=1, 
                               callbacks=[model_checkpoint],max_queue_size=10,
                               workers=3, use_multiprocessing=True)


if __name__ == '__main__':
    train_and_predict()