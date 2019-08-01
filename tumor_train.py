import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import warnings
warnings.filterwarnings("ignore")
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import argparse
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from loss.tumor_loss import *
from augmentation import *
from model.res_net import *
from skimage.transform import resize
import scipy.io as sio
from keras.utils.training_utils import multi_gpu_model 

K.set_image_dim_ordering('tf')

#  global parameters
parser = argparse.ArgumentParser(description='Tumor Training')
#  data folder
parser.add_argument('-data', type=str, default='data/', help='test images')
parser.add_argument('-save_path', type=str, default='Experiments/')
#  other paras
parser.add_argument('-b', type=int, default=40)
parser.add_argument('-input_size', type=int, default=160)
parser.add_argument('-model_weight', type=str, default='weight/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
parser.add_argument('-input_cols', type=int, default=3)

parser.add_argument('-thread_num', type=int, default=20)
args = parser.parse_args()

thread_num = args.thread_num

def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))
limit_mem()


def load_seq_crop_data_masktumor_try(Parameter_List, upper=250, lower=-200):
    img = Parameter_List[0]
    tumor = Parameter_List[1]
    lines = Parameter_List[2]
    numid = Parameter_List[3]
    minindex = Parameter_List[4]
    maxindex = Parameter_List[5]
    pid = Parameter_List[6]
    #  randomly scale
    # scale = np.random.uniform(0.8,1.2)
    scale = np.random.uniform(0.6, 1.0)
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
    window_size=upper-lower
    
    col=0
    for i in range(c - cols // 2, c + cols // 2 + 1):
        sub_img = sio.loadmat(args.data+'image_mat/'+str(pid)+'/'+str(i)+'.mat')['data']
        sub_img[sub_img>upper]=upper
        sub_img[sub_img<lower]=lower
        sub_img -= lower
        cropp_img[:,:,col] = sub_img[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2]
        
        sub_tumor = sio.loadmat(args.data+'label_mat/'+str(pid)+'/'+str(c)+'.mat')['data']
        cropp_tumor[:,:,col] = sub_tumor[a - deps // 2:a + deps // 2, b - rows // 2:b + rows // 2]
        col+=1
    sub_img = gamma(sub_img, window_size)

    # randomly flipping
    flip_num = np.random.randint(0, 8)
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
     
    cropp_img = resize(cropp_img, (args.input_size,args.input_size), order=3, mode='constant', cval=0, clip=True,preserve_range=True)
    cropp_tumor = resize(cropp_tumor, (args.input_size,args.input_size), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    return cropp_img, cropp_tumor[:,:,1]

def generate_arrays_from_file(batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list):
    
    liverlist = [32,34,38,41,47,87,89,91,105,106,114,115,119]
    while 1:
        X = np.zeros((batch_size, args.input_size, args.input_size, args.input_cols), dtype='float32')
        Y = np.zeros((batch_size, args.input_size, args.input_size, 1), dtype='int16')
        Parameter_List = []
        for idx in range(batch_size):

            count = np.random.choice(range(len(trainidx)))
            img = img_list[count]
            tumor = tumor_list[count]
            minindex = minindex_list[count]
            maxindex = maxindex_list[count]
            num = np.random.randint(0,6)
            if (num < 3 ) or (trainidx[count] in liverlist):
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
            X[idx, :, :, :] = result_list[idx][0]
            Y[idx, :, :, 0] = result_list[idx][1]
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
        print (idx)
        img=None
        tumor=None
        img_list.append(img)
        tumor_list.append(tumor)

        maxmin = np.loadtxt(args.data + 'myTrainingDataTxt/LiverBox/box_' + str(idx) + '.txt', delimiter=' ')
        minindex = maxmin[0:3]
        maxindex = maxmin[3:6]
        minindex = np.array(minindex, dtype='int')
        maxindex = np.array(maxindex, dtype='int')
        del maxmin
        slice_amt = len(os.listdir(args.data+'image_mat/'+str(idx)))
        minindex[0] = max(minindex[0] - 3, 0)
        minindex[1] = max(minindex[1] - 3, 0)
        minindex[2] = max(minindex[2] - 3, 0)
        maxindex[0] = min(512, maxindex[0] + 3)
        maxindex[1] = min(512, maxindex[1] + 3)
        maxindex[2] = min(slice_amt, maxindex[2] + 3)
        minindex_list.append(minindex)
        maxindex_list.append(maxindex)
        f1 = open(args.data + 'myTrainingDataTxt/TumorPixels/tumor_' + str(idx) + '.txt', 'r')
        tumorline = f1.readlines()
        tumorlines.append(tumorline)
        tumoridx.append(len(tumorline))
        f1.close()
        f2 = open(args.data + 'myTrainingDataTxt/LiverPixels/liver_' + str(idx) + '.txt', 'r')
        liverline = f2.readlines()
        liverlines.append(liverline)
        liveridx.append(len(liverline))
        f2.close()
        
    return trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list

def train_and_predict():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = ResUNet(drop=0.2)
    model.load_weights(args.model_weight, by_name=True)
    # model = multi_gpu_model(model, gpus=2)
    sgd = SGD(lr=1e-3,momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=[F_score_loss], metrics=[Dice, recall])
    
    trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list = load_fast_files(args)

    print('-'*30)
    print('Fitting model......')
    print('-'*30)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model_checkpoint = ModelCheckpoint(args.save_path + '/model/Res_weights.{epoch:02d}-{Dice:.3f}.h5', 
                                       monitor='loss', mode='min', verbose = 1,
                                       save_best_only=True,save_weights_only=True)

    model.fit_generator(generate_arrays_from_file(args.b, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx,
                                                  liveridx, minindex_list, maxindex_list),steps_per_epoch=100,
                        epochs= 1000, verbose = 1, callbacks = [model_checkpoint], max_queue_size=10,
                        workers=3, use_multiprocessing=True)

    print ('Finised Training .......')

if __name__ == '__main__':
    train_and_predict()