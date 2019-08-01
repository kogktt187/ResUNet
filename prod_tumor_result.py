from model.res_net import *
from medpy.io import load, save
import numpy as np
from scipy import ndimage
from skimage import measure
import keras.backend as K
from skimage.transform import resize
from keras.utils.training_utils import multi_gpu_model
from scipy.special import softmax


model = ResUNet(NUM_CLASS=3)
model.load_weights('weight/tumor_model.h5')
thres_liver = 0.8
thres_tumor = 0.5
upper=250
lower=-200
rescale_size=672
data_path = 'data/'

def get_data(i,data_path,upper=250,lower=-200,rescale_size=672):
    image, i_h = load(data_path+'TestData/test-volume-'+str(i)+'.nii')
    liver, l_h = load(data_path+'livermask/test-segmentation-'+str(i)+'.nii')
    liver[liver==2]=1
    res = np.zeros_like(image, dtype='uint8')
    liver_range = [[min(s), max(s)+1] for s in np.where(liver>0)]
    image = image[:,:,liver_range[2][0]:liver_range[2][1]]
    liver = liver[:,:,liver_range[2][0]:liver_range[2][1]]
    image[image>upper]=upper
    image[image<lower]=lower
    image -=lower

    liver = liver.transpose((2,0,1))
    image = image.transpose((2,0,1))
    if rescale_size!=512: image = resize(image, (image.shape[0],rescale_size, rescale_size), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
    input_im = np.zeros(image.shape+(3,))
    for j in range(input_im.shape[0]):
        if j==0 or j==input_im.shape[0]-1:
            input_im[j,:,:,0] = image[j]
            input_im[j,:,:,1] = image[j]
            input_im[j,:,:,2] = image[j]
        else:
            input_im[j,:,:,0] = image[j-1]
            input_im[j,:,:,1] = image[j]
            input_im[j,:,:,2] = image[j+1]    

    return input_im, liver, l_h, liver_range, res

for idx in range(70):
    print(idx)
    input_im, liver, l_h, liver_range, res = get_data(idx,data_path=data_path)
    
    print('start predict')
    a = time.time()
    pred = model.predict(input_im, batch_size=20)
    pred = softmax(pred, axis=-1)
    time_comsume.append(time.time()-a)
    pred_liver = pred[:,:,:,1].copy()
    pred_tumor = pred[:,:,:,2].copy()
    pred_liver[pred_liver>=thres_liver]=1
    pred_liver[pred_liver<thres_liver]=0
    pred_tumor[pred_tumor>=thres_tumor]=1
    pred_tumor[pred_tumor<thres_tumor]=0

    if rescale_size!=512: pred_tumor = resize(pred_tumor, (pred_tumor.shape[0],512,512), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    pred_tumor[pred_tumor>=0.5]=1
    pred_tumor[pred_tumor<0.5]=0


    liver_dilation = ndimage.binary_dilation(liver, iterations=1).astype(int)
    # combine liver and tumor
    liver_dilation[pred_tumor==1]=1

    box=[]
    [liver_biggest, num] = measure.label(liver_dilation, return_num=True)
    region = measure.regionprops(liver_biggest)
    for i in range(num):
        box.append(region[i].area)
    label_num = box.index(max(box)) + 1
    liver_biggest[liver_biggest != label_num] = 0
    liver_biggest[liver_biggest == label_num] = 1
    liver_biggest = ndimage.binary_fill_holes(liver_biggest).astype(int)
    pred_tumor = pred_tumor * liver_biggest

    # remove small tumor
    # box=[]
    # pred_dilation = ndimage.binary_dilation(pred_tumor, iterations=2).astype(int)
    # [pred_biggest, num] = measure.label(pred_tumor, return_num=True)
    # region = measure.regionprops(pred_biggest)
    # for i in range(num):
    #     box.append(region[i].area)
    #     if region[i].area<200: pred_tumor[pred_biggest == i+1] = 0
    # box.sort()
    # print(box)

    pred_tumor = ndimage.binary_fill_holes(pred_tumor).astype(int)
    print(np.sum(pred_tumor))
    if np.sum(pred_tumor)<50: pred_tumor[pred_tumor>0]=0
    liver[pred_tumor==1]=2
    liver = liver.transpose((1,2,0))
    res[:,:,liver_range[2][0]:liver_range[2][1]] += liver
    save(res, 'result/test-segmentation-'+str(idx)+'.nii', hdr=l_h)

    print('-'*30)