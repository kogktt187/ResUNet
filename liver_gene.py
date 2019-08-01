import cv2
import os
import nibabel as nib
import scipy.io as sio
import numpy as np
import keras
import keras.backend as K
from skimage.transform import resize
import random
from multiprocessing.dummy import Pool as ThreadPool
root_folder = os.getcwd()


def gamma(img, window_size):
    img = np.float32(img)
    g = np.random.uniform(0.5,2.0)
    img/=window_size
    img = (img**g)*window_size
    return img

def get_img_label(person_id, slice_id, data_path, lower_HU=-150, upper_HU=250):
    window_size=upper_HU-lower_HU
    person_id = str(person_id)
    slice_id = str(slice_id)
    image_dir = os.path.join(data_path, 'image_mat')
    label_dir = os.path.join(data_path, 'label_mat')

    image_person = os.path.join(image_dir, person_id)
    label_person = os.path.join(label_dir, person_id)  
    
    image_path = os.path.join(image_person, slice_id+'.mat')
    label_path = os.path.join(label_person, slice_id+'.mat')

    img = sio.loadmat(image_path)['data']
    liver = sio.loadmat(label_path)['data']
    img[img<lower_HU] = lower_HU
    img[img>upper_HU] = upper_HU
    img-=lower_HU
    img = np.expand_dims(img, axis=-1)
    liver[liver==2]=1
    liver = np.expand_dims(liver, axis=-1)
    return img, liver

def get_liver_3pic_gene(Parameter_List):
    person_id = Parameter_List[0] 
    final_slice = Parameter_List[1]
    slice_id = int(Parameter_List[2])
    IMG_SIZE = Parameter_List[3]
    lower_HU = Parameter_List[4]
    upper_HU = Parameter_List[5]
    data_path = Parameter_List[6]
    window_size=upper_HU-lower_HU

    if slice_id==0 or slice_id==final_slice:
        img, liver = get_img_label(person_id,slice_id,lower_HU=lower_HU,upper_HU=upper_HU, data_path=data_path)
        img = np.concatenate((img, img, img), axis=-1)
        liver = np.concatenate((liver, liver, liver), axis=-1)
    else:
        img_1, liver_1 = get_img_label(person_id, slice_id-1,lower_HU=lower_HU,upper_HU=upper_HU, data_path=data_path)
        img_2, liver_2 = get_img_label(person_id, slice_id,lower_HU=lower_HU,upper_HU=upper_HU, data_path=data_path)
        img_3, liver_3 = get_img_label(person_id, slice_id+1,lower_HU=lower_HU,upper_HU=upper_HU, data_path=data_path)
        img = np.concatenate((img_1, img_2, img_3), axis=-1)
        liver = np.concatenate((liver_1, liver_2, liver_3), axis=-1)
    img = gamma(img, window_size=window_size)

    # flip augment
    flip_num = np.random.randint(0, 8)
    if flip_num == 1:
        img = np.flipud(img)
        liver = np.flipud(liver)
    elif flip_num == 2:
        img = np.fliplr(img)
        liver = np.fliplr(liver)
    elif flip_num == 3:
        img = np.rot90(img, k=1, axes=(1, 0))
        liver = np.rot90(liver, k=1, axes=(1, 0))
    elif flip_num == 4:
        img = np.rot90(img, k=3, axes=(1, 0))
        liver = np.rot90(liver, k=3, axes=(1, 0))
    elif flip_num == 5:
        img = np.fliplr(img)
        liver = np.fliplr(liver)
        img = np.rot90(img, k=1, axes=(1, 0))
        liver = np.rot90(liver, k=1, axes=(1, 0))
    elif flip_num == 6:
        img = np.fliplr(img)
        liver = np.fliplr(liver)
        img = np.rot90(img, k=3, axes=(1, 0))
        liver = np.rot90(liver, k=3, axes=(1, 0))
    elif flip_num == 7:
        img = np.flipud(img)
        liver = np.flipud(liver)
        img = np.fliplr(img)
        liver = np.fliplr(liver)

    img = resize(img, (IMG_SIZE, IMG_SIZE), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
    liver = resize(liver, (IMG_SIZE, IMG_SIZE), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    return img, liver[:,:,1]

def liver_generator(dataset_info, data_path, thread_num=14, IMG_SIZE=224, lower_HU=-150, upper_HU=250, batch_size=32):

    training_idx = list(range(len(dataset_info)))
    while True:
        X = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3))
        y = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 1))
        Parameter_List = []
        for i in range(batch_size):
            idx=np.random.choice(training_idx)
            person_id=dataset_info[idx][0]
            final_slice=dataset_info[idx][1]
            a = np.random.randint(0,8)
            if a<5: sid = np.random.choice(dataset_info[idx][2])
            else: sid = np.random.randint(dataset_info[idx][1]+1)
            Parameter_List.append([person_id, final_slice, sid, IMG_SIZE, lower_HU, upper_HU, data_path])
        pool = ThreadPool(thread_num)
        result_list = pool.map(get_liver_3pic_gene, Parameter_List)
        pool.close()
        pool.join()
        for j in range(len(result_list)):
            X[j,:,:,:] = result_list[j][0]
            y[j,:,:,0] = result_list[j][1]
        yield (X, y)

###

