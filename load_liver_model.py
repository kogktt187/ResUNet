from medpy.io import load, save
import numpy as np
import cv2
from scipy import ndimage
from skimage import measure
from scipy.special import softmax
import keras.backend as K
from skimage.transform import resize
from model.res_net import *
import os
from evaluate import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


NUM_CLASS=1
model = ResUNet(NUM_CLASS=NUM_CLASS)
model.load_weights('weight/liver_model.h5')
thre = 0.9
upper=250
lower=-200
rescale_size=256
data_path = 'data/'

for i in range(131):
    print(i)
    label, l_h = load(data_path+'label/segmentation-'+str(i)+'.nii')
    image, i_h = load(data_path+'image/volume-'+str(i)+'.nii')
    image = image.transpose((2,0,1))
    label = label.transpose((2,0,1))
    
    liver = label.copy()
    liver[liver==2]=1

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
            
    pred = model.predict(img, batch_size=64)
    pred_liver = pred[:,:,:,NUM_CLASS-1]
    pred_liver[pred_liver>=thre]=1
    pred_liver[pred_liver<thre]=0
    pred_liver = ndimage.binary_fill_holes(pred_liver).astype(float)
    
    # preserve the largest liver
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
    
    # show evaluations
    D = Dice(liver, pred_liver)
    p = precision(liver, pred_liver)
    r = recall(liver, pred_liver)
    print(D)
    print('p:', p)
    print('r:', r)
    print('-'*30)

