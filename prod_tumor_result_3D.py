from medpy.io import load, save
import numpy as np
from scipy import ndimage
from skimage import measure
from model.hybrid_res50_unet3d import hybrid
from model.hybrid_res50_unet3d_3pic import hybrid_3pic
from model.U_net3D import *
from keras.utils.training_utils import multi_gpu_model
import argparse
from skimage.transform import resize
from scipy.special import softmax
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='product 3D result')
#  data folder
parser.add_argument('-data', type=str, default='data/', help='test images')
parser.add_argument('-save_path', type=str, default='Experiments/')
#  other paras
parser.add_argument('-b', type=int, default=1)
parser.add_argument('-input_size', type=int, default=640)
parser.add_argument('-model_weight', type=str, default='./model/model_best.hdf5')
parser.add_argument('-input_cols', type=int, default=8)
args = parser.parse_args()
cols = args.input_cols
window = cols//2


model = hybrid_3pic(args)
model.load_weights('Experiments/H_model/weights.07-0.932.h5')

thres_liver = 0.7
thres_tumor = 0.5
rescale_size=args.input_size
time_comsume = []
upper=250
lower=-200
def get_data(i):
    image, i_h = load(args.data+'TestData/test-volume-'+str(i)+'.nii')
    liver, l_h = load(args.data+'livermask/test-segmentation-'+str(i)+'.nii')
    liver[liver>1]=1
    res = np.zeros_like(image, dtype='uint8')
    liver_range = [[min(s), max(s)+1] for s in np.where(liver>0)]
    liver = liver[:,:,liver_range[2][0]:liver_range[2][1]]
    if liver_range[2][0]==0:
        image = image[:,:,liver_range[2][0]:liver_range[2][1]+1]
        single_slice = np.ones((512,512,1))
        image = np.concatenate((single_slice, image), axis=-1)
    else:
        image = image[:,:,liver_range[2][0]-1:liver_range[2][1]+1]

    image = image.transpose((2,0,1))
    liver = liver.transpose((2,0,1))

    return image, liver, l_h, liver_range, res

for idx in range(70):

    print(idx)
    image, liver, l_h, liver_range, res = get_data(idx)
    print(image.shape)
    
    image[image>upper]=upper
    image[image<lower]=lower
    image-=lower
    if rescale_size!=512: image=resize(image, (image.shape[0],rescale_size,rescale_size), order=3, mode='constant', cval=0, 
                                       clip=True,preserve_range=True)
    img = np.expand_dims(image, axis=-1)
    img_3D = []

    flag=True
    for j in range(0, img.shape[0],window):
        if j+cols>img.shape[0]:
            if flag:
                img_3D.append(img[-cols:])
                flag=False
            else: pass
        else:
            img_3D.append(img[j:j+cols])
    img_3D = np.array(img_3D)
    img_3D = img_3D.transpose((0,2,3,1,4))
    print(img_3D.shape)
    print('start predict')
    a = time.time()
    pred = model.predict(img_3D, batch_size=args.b)
    pred = softmax(pred, axis=-1)
    pred_liver_3D = pred[:,:,:,:,1].copy()
    pred_tumor_3D = pred[:,:,:,:,2].copy()
    interval = time.time()-a
    time_comsume.append(interval)
    print(pred_tumor_3D.shape)

#   stack pred result
    pred_tumor = np.zeros((rescale_size,rescale_size,liver_range[2][1]-liver_range[2][0]+2))
    pred_liver = np.zeros((rescale_size,rescale_size,liver_range[2][1]-liver_range[2][0]+2))
    pred_count = np.zeros((rescale_size,rescale_size,liver_range[2][1]-liver_range[2][0]+2))
    for j in range(pred_tumor_3D.shape[0]):
        if j==pred.shape[0]-1: 
            pred_tumor[:,:,-cols+1:-1] += pred_tumor_3D[j,:,:,1:-1]
            pred_liver[:,:,-cols+1:-1] += pred_liver_3D[j,:,:,1:-1]
            pred_count[:,:,-cols+1:-1] += np.ones((rescale_size,rescale_size,cols-2))
        else: 
            pred_tumor[:,:,j*window+1:j*window+cols-1] += pred_tumor_3D[j,:,:,1:-1]
            pred_liver[:,:,j*window+1:j*window+cols-1] += pred_liver_3D[j,:,:,1:-1]
            pred_count[:,:,j*window+1:j*window+cols-1] += np.ones((rescale_size,rescale_size,cols-2))
    pred_count += 1e-5
    pred_tumor /= pred_count
    pred_liver[pred_liver>=thres_liver]=1
    pred_liver[pred_liver<thres_liver]=0
    pred_tumor[pred_tumor>=thres_tumor]=1
    pred_tumor[pred_tumor<thres_tumor]=0
    pred_tumor = pred_tumor[:,:,1:-1]
    if rescale_size!=512: pred_tumor=resize(pred_tumor, (512,512), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    pred_tumor[pred_tumor>=0.5]=1
    pred_tumor[pred_tumor<0.5]=0
    
#   preserve the largest liver
    liver = liver.transpose((1,2,0)) 
    liver_dilation = ndimage.binary_dilation(liver, iterations=2).astype(int)
    liver_dilation[pred_tumor==1]=1
    
    box = []
    [liver_labels, num] = measure.label(liver_dilation, return_num=True)
    region = measure.regionprops(liver_labels)
    for i in range(num):
        box.append(region[i].area)
    label_num = box.index(max(box)) + 1
    liver_labels[liver_labels != label_num] = 0
    liver_labels[liver_labels == label_num] = 1
    liver_labels = ndimage.binary_fill_holes(liver_labels).astype(int)
    pred_tumor*=liver_labels

    # remove small tumor
    # box=[]
    # tumor_dilation = ndimage.binary_dilation(pred_tumor, iterations=1).astype(int)
    # [pred_biggest, num] = measure.label(tumor_dilation, return_num=True)
    # region = measure.regionprops(pred_biggest)
    # for i in range(num):
    #     box.append(region[i].area)
    #     if region[i].area<=50: pred_tumor[pred_biggest == i+1] = 0
    # box.sort()
    # print(box)

#   output result
    print(np.sum(pred_tumor))
    liver[pred_tumor==1]=2
    res[:,:,liver_range[2][0]:liver_range[2][1]] += liver
    save(res, 'result/test-segmentation-'+str(idx)+'.nii', hdr=l_h)
    print('-'*30)

for i in time_comsume:
    print(i)
s=''
f = open('time.txt','w')
for i in time_comsume:
    s+=(str(i)+'\n')
f.write(s)