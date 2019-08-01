import numpy as np
def HE_bright(image, window_size=600, factor=1):
#     factor= np.random.uniform(1,1.5)
    image = image.astype('int16')
    img = image.copy()
    img = img[img>0]
    img = img[img<window_size]
    hist,bins = np.histogram(img.flatten(),window_size+1,[0,window_size+1])
    cdf = hist.cumsum()
    
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = window_size * (((cdf_m - cdf_m.min())/(cdf_m.max()-cdf_m.min()))**factor)-100
    cdf = np.ma.filled(cdf_m,0).astype('int16')
    cdf[cdf<0]=0
    image=cdf[image]
    return image

def HE_dark(image, window_size=600, factor=1):
#     factor= np.random.uniform(0.6,1)
    image = image.astype('int16')
    img = image.copy()
    img = img[img>0]
    img = img[img<window_size]
    hist,bins = np.histogram(img.flatten(),window_size+1,[0,window_size+1])
    cdf = np.flip(np.flip(hist).cumsum())
    
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = window_size * (((cdf_m - cdf_m.min())/(cdf_m.max()-cdf_m.min()))**factor)+100
    cdf = np.ma.filled(cdf_m,0).astype('int16')
    cdf = np.flip(cdf)
    cdf[cdf>window_size]=window_size
    image=cdf[image]
    return image

def gamma(cropp_img, window_size=450):
    # randomly constract
    gamma = np.random.uniform(0.4,2.5)
    cropp_img = cropp_img.astype('float32')
    cropp_img /=window_size
    cropp_img = (cropp_img**gamma)*window_size
    return cropp_img