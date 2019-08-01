from medpy.io import load, save
import numpy as np
import cv2
from scipy import ndimage
from skimage import measure
import keras.backend as K
from scipy.special import softmax
from skimage.transform import resize
from model.res_net import *
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from evaluate import *



data_path = '../data/'
model = ResUNet(NUM_CLASS=2)
model.load_weights('weight/liver_model.h5')

thre = 0.9
upper=250
lower=-200
rescale_size=256

for idx in range(70):
    print(idx)
    image, i_h = load(data_path+'TestData/test-volume-'+str(idx)+'.nii')
    image = image.transpose((2,0,1))
    image[image>upper]=upper
    image[image<lower]=lower
    image-=lower
    if rescale_size!=512: image = resize(image, (image.shape[0],rescale_size,rescale_size), order=3, mode='constant', 
                                         cval=0, clip=True,preserve_range=True)
    
    img = np.zeros((image.shape+(3,)))
    for j in range(image.shape[0]):
        if j==0 or j==image.shape[0]-1:
            img[j,:,:,0],img[j,:,:,1],img[j,:,:,2] = image[j],image[j],image[j]
        else:
            img[j,:,:,0] = image[j-1]
            img[j,:,:,1] = image[j]
            img[j,:,:,2] = image[j+1]
            
    pred = model.predict(img, batch_size=128)
    pred_liver = pred[:,:,:,0]
    pred_liver[pred_liver>=thre]=1
    pred_liver[pred_liver<thre]=0
    
    ### preserve the largest liver
    box=[]
    liver_dilation = pred_liver
    # liver_dilation = ndimage.binary_dilation(pred_liver, iterations=1).astype(int)
    [liver_biggest, num] = measure.label(liver_dilation, return_num=True)
    region = measure.regionprops(liver_biggest)
    for i in range(num):
        box.append(region[i].area)
    label_num = box.index(max(box)) + 1
    pred_liver[liver_biggest != label_num] = 0
    pred_liver[liver_biggest == label_num] = 1 

    if rescale_size!=512: pred_liver = resize(pred_liver, (pred_liver.shape[0],512,512), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    pred_liver[pred_liver>=0.5]=1
    pred_liver[pred_liver<0.5]=0
    pred_liver = pred_liver.transpose((1,2,0))
    pred_liver = np.int16(pred_liver)
    pred_liver = np.uint8(pred_liver)
    print(pred_liver.shape)
    save(pred_liver, data_path+'livermask/test-segmentation-'+str(idx)+'.nii', hdr=i_h)

