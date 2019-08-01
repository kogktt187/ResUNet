from medpy.io import load, save
import numpy as np
from scipy import ndimage
from skimage import measure
from model.hybrid_res50_unet3d import *
from model.hybrid_res50_unet3d_3pic import *
from keras.utils.training_utils import multi_gpu_model
from skimage.transform import resize
import argparse
from scipy.special import softmax
from evaluate import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='load tumor 3D')
#  data folder
parser.add_argument('-data', type=str, default='data/', help='test images')
parser.add_argument('-save_path', type=str, default='Experiments/')
#  other paras
parser.add_argument('-b', type=int, default=1)
parser.add_argument('-input_size', type=int, default=672)
parser.add_argument('-input_cols', type=int, default=8)
args = parser.parse_args()


def get_data(idx):
    print(idx)
    label, l_h = load(args.data+'label/segmentation-'+str(idx)+'.nii')
    image, i_h = load(args.data+'image/volume-'+str(idx)+'.nii')
    res = np.zeros_like(image, dtype='uint8')
    liver_range = [[min(i), max(i)+1] for i in np.where(label>0)]
    label = label[:,:,liver_range[2][0]:liver_range[2][1]]      
    if liver_range[2][0]==0:
        image = image[:,:,liver_range[2][0]:liver_range[2][1]+1]
        single_slice = np.ones((512,512,1))
        image = np.concatenate((single_slice, image), axis=-1)
    else:
        image = image[:,:,liver_range[2][0]-1:liver_range[2][1]+1]
    image = image.transpose((2,0,1))
    label = label.transpose((2,0,1))

    liver = label.copy()
    liver[liver==2]=1
    tumor = label-liver
    
    return image, liver, tumor, liver_range, res

cols = args.input_cols
window = cols//2
model = hybrid_3pic(args)
model.load_weights('weight/tumor_model_3D.h5')

thres_liver=0.5
thres_tumor=0.5
rescale_size=args.input_size
upper=250
lower=-200

for i in range(131):

    image, liver, tumor, liver_range, res = get_data(i)
    
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
    print('===start predict===')
    pred = model.predict(img_3D, batch_size=1)
    pred_liver_3D = pred[:,:,:,:,1].copy()
    pred_tumor_3D = pred[:,:,:,:,2].copy()

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
    pred_tumor = pred_tumor.transpose((2,0,1))
    if rescale_size!=512: pred_tumor=resize(pred_tumor, (pred_tumor.shape[0],512,512), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    pred_tumor[pred_tumor>=0.5]=1
    pred_tumor[pred_tumor<0.5]=0
    pred_tumor*=liver

    # box=[]
    # # pred_dilation = ndimage.binary_dilation(pred, iterations=1).astype(int)
    # [pred_biggest, num] = measure.label(pred_tumor, return_num=True)
    # region = measure.regionprops(pred_biggest)
    # for i in range(num):
    #     box.append(region[i].area)
    #     if region[i].area<50: pred_tumor[pred_biggest == i+1] = 0
    # box.sort()
    # print(box)


    D = Dice(pred_tumor, tumor)
    p = precision(tumor, pred_tumor)
    r = recall(tumor, pred_tumor)
    print(D)
    print('p:', p)
    print('r:', r)

    print(np.sum(pred_tumor))
    print('-'*30)