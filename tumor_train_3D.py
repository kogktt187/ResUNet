from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from model.hybrid_res50_unet3d_3pic import hybrid_3pic
from model.U_net3D import *
import keras.backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from loss.tumor_loss_3D import *
from skimage.transform import resize
import argparse
K.set_image_dim_ordering('tf')

import scipy.io as sio
from keras.utils.training_utils import multi_gpu_model 
from scipy import ndimage
from skimage import measure

#  global parameters
parser = argparse.ArgumentParser(description='Keras DenseUnet Training')
#  data folder
parser.add_argument('-data', type=str, default='data/', help='test images')
parser.add_argument('-save_path', type=str, default='Experiments/')
#  other paras
parser.add_argument('-b', type=int, default=1)
parser.add_argument('-input_size', type=int, default=128)
parser.add_argument('-model_weight', type=str, default='./model/model_best.hdf5')
parser.add_argument('-input_cols', type=int, default=8)
parser.add_argument('-arch', type=str, default='3dpart')

#  data augment
parser.add_argument('-mean', type=int, default=48)
args = parser.parse_args()

thread_num = 14
liverlist = [32,34,38,41,47,87,89,91,105,106,114,115,119]


def load_seq_crop_data_masktumor_try(Parameter_List, upper=250, lower=-200):
    img = Parameter_List[0]
    tumor = Parameter_List[1]
    lines = Parameter_List[2]
    numid = Parameter_List[3]
    minindex = Parameter_List[4]
    maxindex = Parameter_List[5]
    pid = Parameter_List[6]
    #  randomly scale
    scale = np.random.uniform(0.8,1.2)

    deps = int(args.input_size * scale)
    rows = int(args.input_size * scale)
    cols = args.input_cols

    sed = np.random.randint(1,numid)
    cen = lines[sed-1]
    cen = np.fromstring(cen, dtype=int, sep=' ')

    a = min(max(minindex[0] + deps//2, cen[0]), maxindex[0]- deps//2-1)
    b = min(max(minindex[1] + rows//2, cen[1]), maxindex[1]- rows//2-1)
    c = min(max(minindex[2] + cols//2, cen[2]), maxindex[2]- cols//2-1)
    cropp_img = np.zeros(((a + deps // 2)-(a - deps // 2), (b + rows // 2)-(b - rows // 2), cols))
    cropp_tumor = np.zeros(((a + deps // 2)-(a - deps // 2), (b + rows // 2)-(b - rows // 2), cols))
    
    col=0
    for i in range(c - cols // 2, c + cols // 2):
        sub_img = sio.loadmat(args.data+'image_mat/'+str(pid)+'/'+str(i)+'.mat')['data']
        sub_img[sub_img>upper]=upper
        sub_img[sub_img<lower]=lower
        sub_img-=lower
        cropp_img[:,:,col] = sub_img[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2]

        sub_tumor = sio.loadmat(args.data+'label_mat/'+str(pid)+'/'+str(i)+'.mat')['data']
        cropp_tumor[:,:,col] = sub_tumor[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2]
        col+=1
        
    # randomly constract
    gamma = np.random.uniform(0.4,2.5)
    cropp_img = cropp_img.astype('float32')
    cropp_img /=450
    cropp_img = (cropp_img**gamma)*450

    # randomly flipping
    flip_num = np.random.randint(0,8)
    if flip_num == 1:
        cropp_img = np.flipud(cropp_img)
        cropp_tumor = np.flipud(cropp_tumor)
    elif flip_num == 2:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
    elif flip_num == 3:
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
    elif flip_num == 4:
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
    elif flip_num == 5:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
    elif flip_num == 6:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
    elif flip_num == 7:
        cropp_img = np.flipud(cropp_img)
        cropp_tumor = np.flipud(cropp_tumor)
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)

    cropp_tumor = resize(cropp_tumor, (args.input_size,args.input_size, args.input_cols), order=0, mode='edge', cval=0, 
                         clip=True, preserve_range=True)
    cropp_img   = resize(cropp_img, (args.input_size,args.input_size, args.input_cols), order=3, mode='constant', cval=0, 
                         clip=True, preserve_range=True)
    return cropp_img, cropp_tumor

def generate_arrays_from_file(batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list):
    while 1:
        X = np.zeros((batch_size, args.input_size, args.input_size, args.input_cols,1), dtype='float32')
        Y = np.zeros((batch_size, args.input_size, args.input_size, args.input_cols,1), dtype='int16')
        Parameter_List = []
        for idx in range(batch_size):
            count = np.random.choice(range(len(trainidx)))
            img = img_list[count]
            tumor = tumor_list[count]
            minindex = minindex_list[count]
            maxindex = maxindex_list[count]
            num = np.random.randint(0,6)
            if num < 3 or (trainidx[count] in liverlist):
                lines = liverlines[count]
                numid = liveridx[count]
            else:
                lines = tumorlines[count]
                numid = tumoridx[count]
            Parameter_List.append([img, tumor, lines, numid, minindex, maxindex, trainidx[count]])
        pool = ThreadPool(thread_num)
        result_list = pool.map(load_seq_crop_data_masktumor_try, Parameter_List)
        pool.close()
        pool.join()
        for idx in range(len(result_list)):
            X[idx, :, :, :, 0] = result_list[idx][0]
            Y[idx, :, :, :, 0] = result_list[idx][1]
        if np.sum(Y==0)==0:
            continue
        if np.sum(Y==1)==0:
            continue
        if np.sum(Y==2)==0:
            continue
        yield (X,Y)

def load_fast_files(args):

    trainidx = list(range(131))
    img_list = []
    tumor_list = []
    minindex_list = []
    maxindex_list = []
    tumorlines = []
    tumoridx = []
    liveridx = []
    liverlines = []
    for idx in trainidx:
        print(idx)
        img = None
        tumor = None
        img_list.append(img)
        tumor_list.append(tumor)

        maxmin = np.loadtxt(args.data+'myTrainingDataTxt/LiverBox/box_' + str(idx) + '.txt', delimiter=' ')
        minindex = maxmin[0:3]
        maxindex = maxmin[3:6]
        minindex = np.array(minindex, dtype='int')
        maxindex = np.array(maxindex, dtype='int')
        minindex[0] = max(minindex[0]-3, 0)
        minindex[1] = max(minindex[1]-3, 0)
        minindex[2] = max(minindex[2]-3, 0)


        slice_amt = len(os.listdir(args.data+'image_mat/'+str(idx)))
        maxindex[0] = min(512, maxindex[0] + 3)
        maxindex[1] = min(512, maxindex[1] + 3)
        maxindex[2] = min(slice_amt, maxindex[2] + 3)
        minindex_list.append(minindex)
        maxindex_list.append(maxindex)

        f1 = open(args.data+ 'myTrainingDataTxt/TumorPixels/tumor_' + str(idx) + '.txt','r')
        tumorline = f1.readlines()
        tumorlines.append(tumorline)
        tumoridx.append(len(tumorline))
        f1.close()

        f2 = open(args.data+ '/myTrainingDataTxt/LiverPixels/liver_' + str(idx) + '.txt','r')
        liverline = f2.readlines()
        liverlines.append(liverline)
        liveridx.append(len(liverline))
        f2.close()
    return trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list

def train_and_predict(args):

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = hybrid_3pic(args, drop=0.2)
    # model = multi_gpu_model(model, gpus=2)
    model.load_weights('weight/tumor_model.h5', by_name=True)
    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=-True)
    model.compile(optimizer=sgd, loss=[F_score_loss], metrics=[Dice, recall])

    
    trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list = load_fast_files(args)

    model_checkpoint = ModelCheckpoint(args.save_path + 'H_model/weights.{epoch:02d}-{Dice:.3f}.h5', monitor='loss', verbose = 2,
                                       save_best_only=True,save_weights_only=True,mode = 'min', period = 1)
    print('-'*30)
    print('Fitting model......')
    print('-'*30)

    model.fit_generator(generate_arrays_from_file(args.b, trainidx, img_list, tumor_list, tumorlines, liverlines,
                                                  tumoridx, liveridx, minindex_list, maxindex_list),
                        steps_per_epoch=100, epochs= 3000, verbose = 1, 
                        callbacks = [model_checkpoint], max_queue_size=10,
                        workers=3, use_multiprocessing=True)
    print ('Finised Training .......')

if __name__ == '__main__':
    train_and_predict(args)
