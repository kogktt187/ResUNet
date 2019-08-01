from medpy.io import load, save
import os
import os.path
import numpy as np
import scipy.io as sio

def generate_mat(image_path, label_path, save_img_folder, save_label_folder):
    if not os.path.exists(save_img_folder): os.mkdir(save_img_folder)
    if not os.path.exists(save_label_folder): os.mkdir(save_label_folder)
    
    for i in range(131):
        print(i)
        person_img_path = save_img_folder+str(i)+'/'
        person_label_path = save_label_folder+str(i)+'/'
        if not os.path.exists(person_img_path): os.mkdir(person_img_path)
        if not os.path.exists(person_label_path): os.mkdir(person_label_path)
        
        image, h = load(image_path+'volume-'+str(i)+'.nii')
        label, h = load(label_path+'segmentation-'+str(i)+'.nii')
        image = image.astype('int16')
        label = label.astype('uint8')
        for j in range(image.shape[2]):
            img_mat_name = os.path.join(person_img_path, str(j)+'.mat')
            label_mat_name = os.path.join(person_label_path, str(j)+'.mat')
            sio.savemat(img_mat_name, {'data': image[:,:,j]})
            sio.savemat(label_mat_name, {'data': label[:,:,j]})

def generate_liverslice_txt(label_path, save_folder):
    if not os.path.exists(save_folder): os.mkdir(save_folder)

    for i in range(131):
        print(i)
        livertumor, header = load(label_path+'segmentation-'+str(i)+'.nii')
        f = open(save_folder+ str(i) + '.txt', 'w')
        index = np.unique(np.where(livertumor>0)[2])
        for i in index:
            f.write(str(i)+"\n")
        f.close()

def generate_livertxt(label_path, save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Generate Livertxt
    if not os.path.exists(save_folder+'LiverPixels'):
        os.mkdir(save_folder+'LiverPixels')

    for i in range(0,131):
        print(i)
        livertumor, header = load(label_path+'segmentation-'+str(i)+'.nii')
        f = open(save_folder+'/LiverPixels/liver_' + str(i) + '.txt', 'w')
        index = np.where(livertumor==1)
        x = index[0]
        y = index[1]
        z = index[2]
        np.savetxt(f, np.c_[x,y,z], fmt="%d")
	
        f.write("\n")
        f.close()

def generate_tumortxt(label_path, save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Generate Livertxt
    if not os.path.exists(save_folder+'TumorPixels'):
        os.mkdir(save_folder+'TumorPixels')

    for i in range(0,131):
        print(i)
        livertumor, header = load(label_path+'segmentation-'+str(i)+'.nii')
        f = open(save_folder+"/TumorPixels/tumor_"+str(i)+'.txt','w')
        index = np.where(livertumor==2)

        x = index[0]
        y = index[1]
        z = index[2]

        np.savetxt(f,np.c_[x,y,z],fmt="%d")

        f.write("\n")
        f.close()

def generate_txt(label_path, save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Generate Livertxt
    if not os.path.exists(save_folder+'LiverBox'):
        os.mkdir(save_folder+'LiverBox')
    for i in range(0,131):
        print(i)
        values = np.loadtxt(save_folder+'LiverPixels/liver_' + str(i) + '.txt', delimiter=' ', usecols=[0, 1, 2])
        a = np.min(values, axis=0)
        b = np.max(values, axis=0)
        box = np.append(a,b, axis=0)
        np.savetxt(save_folder+'LiverBox/box_'+str(i)+'.txt', box,fmt='%d')



data_path = 'data/'

print ("Generate slice mat ")
generate_mat(image_path=data_path+'image/', label_path=data_path+'label/', 
             save_img_folder=data_path+'image_mat/', save_label_folder=data_path+'label_mat/')

print ("Generate liver slice txt ")
generate_liverslice_txt(label_path=data_path+'label/', save_folder=data_path+'liver_slice/')

print ("Generate liver txt ")
generate_livertxt(label_path=data_path+'label/', save_folder=data_path+'myTrainingDataTxt/')

print ("Generate tumor txt")
generate_tumortxt(label_path=data_path+'label/', save_folder=data_path+'myTrainingDataTxt/')

print ("Generate liver box ")
generate_txt(label_path=data_path+'label/', save_folder=data_path+'myTrainingDataTxt/')
