from medpy.io import load, save
import numpy as np
from scipy import ndimage
from skimage import measure
from model.res_net import *
from scipy.special import softmax
from skimage.transform import resize
from evaluate import *

def get_data(idx, rescale_size=512,data_path='data/',upper=250,lower=-200):
    image, i_h = load(data_path+'image/volume-'+str(idx)+'.nii')
    label, l_h = load(data_path+'label/segmentation-'+str(idx)+'.nii')
    label = label.transpose((2,0,1))
    image = image.transpose((2,0,1))
    liver_range = [[min(s), max(s)+1] for s in np.where(label>0)]
    image = image[liver_range[0][0]:liver_range[0][1]]
    label = label[liver_range[0][0]:liver_range[0][1]]
    liver = np.int16(label>0)
    tumor = label-liver
    
    image[image>upper]=upper
    image[image<lower]=lower
    image-=lower
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
    return input_im, liver, tumor, liver_range


model = ResUNet()
model.load_weights('weight/tumor_model.h5')

non_tumor_list = [32,34,38,41,47,87,89,91,105,106,114,115,119]
small_tumor = [5,73,83]
bright_tumor = [45,121,39,54]
test_case = small_tumor+bright_tumor+non_tumor_list
thre = 0.9
rescale_size = 672
data_path = 'data/'
for i in range(131):

    print(i)
    input_im, liver, tumor, liver_range = get_data(i,rescale_size=rescale_size,data_path=data_path,upper=250,lower=-200)
    pred = model.predict(input_im, batch_size=20)
    pred[pred>=thre]=1
    pred[pred<thre]=0
    pred = pred[:,:,:,2]
    if rescale_size!=512: pred = resize(pred, (pred.shape[0],512,512), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    pred[pred>=0.5]=1
    pred[pred<0.5]=0
    pred*=liver

    # show evaluations
    D = Dice(pred, tumor)
    p = precision(tumor, pred)
    r = recall(tumor, pred)
    print(D)
    print('p:', p)
    print('r:', r)
    print(np.sum(pred))
    print('-'*30)
