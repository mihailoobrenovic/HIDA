#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import config 
import warnings
from scipy.ndimage.filters import gaussian_filter



#####################stats##########################


ms_eurosat_stats = {"mean":[1353.73046875, 1117.2020263671875, 1041.8876953125, 946.5513305664062, 1199.1883544921875, 2003.0101318359375,
                            2374.01171875, 2301.222412109375, 732.1828002929688, 12.099513053894043, 1820.6893310546875, 
                            1118.1998291015625, 2599.784912109375],
                    "std": [30.343395233154297, 66.4549560546875, 71.52734375, 86.9700698852539, 70.47565460205078, 81.35286712646484, 
                            97.88168334960938, 99.96805572509766, 27.891748428344727, 0.32882159948349, 92.60734558105469, 
                            87.39993286132812, 106.57888793945312],
                    "clip": np.array([2019, 2508, 2508, 2508, 2534, 3674, 4604, 4577, 1693, 20, 3769, 2786, 5024])}


resisc_stats = {"mean" :[84.92513592, 90.46876268, 79.33871272], "std": [3.23005124, 2.9013181 , 3.05717543]}







#####################functions#########################
def standardizeIMG(img, mean, std):
    """Standardize a array image(channel last) with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will standardize each channel of the input
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        img : array image(channel last)
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        
    Returns:
        a standardized copy of img
    
    Raises:
        ValueError if len(mean) != imgShape[-1]) or (len(std) != imgShape[-1])
    """
    
    imgShape = img.shape
    
    if (len(mean) != imgShape[-1]) or (len(std) != imgShape[-1]):
        raise ValueError("length of mean and std should be equal to the number of channels of img")
        
    else:
        img_r = np.zeros(imgShape, dtype=np.float32)
        for i, (meani, stdi)  in enumerate(zip(mean, std)):
            img_r[:, :, i] = (img[:, :, i] - meani)/stdi
            
        return img_r
    
    
    
    
    
def unstandardizeIMG(img, mean, std):
    """reverse standardization transformation of a array image(channel last) with 
    mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this operation will be apply on each channel of the input
    ``output[channel] =  (input[channel] * std[channel]) + mean[channel]``

    Args:
        img : array image(channel last)
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        
    Returns:
        a unstandardized copy of img
    
    Raises:
        ValueError if len(mean) != imgShape[-1]) or (len(std) != imgShape[-1])
    """
    
    imgShape = img.shape
    
    if (len(mean) != imgShape[-1]) or (len(std) != imgShape[-1]):
        raise ValueError("length of mean and std should be equal to the number of channels of img")
        
    else:
        img_r = np.zeros(imgShape)
        for i, (meani, stdi)  in enumerate(zip(mean, std)):
            img_r[:, :, i] = (img[:, :, i] * stdi) + meani
            
        return img_r
    
    



def normalizeIMG(img):
    """
    Computes the per channel normalization of the input image
    output[channel] = (input[channel] - min(input[channel]))/ (max(input[channel]) - min(input[channel]))
    Parameters
    ----------
    img : a array image(channel last)

    Returns
    -------
    img_r : a normalized copy of img

    """
    imgShape = img.shape
    img_r = np.zeros(imgShape)
    
    for i in range(imgShape[-1]):
        img_r[:, :, i] = (img[:, :, i]  - img[:, :, i].min()) / (img[:, :, i].max()  - img[:, :, i].min())
    
    return img_r






##################################################################



#process for RGB img
def processingRGB(x):
    
    if config.processing_type == 'normalization':
        x = x/255.0
        # x = normalizeIMG(x)
        
        
        
    # elif config.processing_type == 'standardization' and config.staining_s == 'resisc':
    elif config.processing_type == 'standardization':
        mean = resisc_stats["mean"]
        std  = resisc_stats["std"]
        x = standardizeIMG(x, mean, std)
        
    return x



#process for MS img
def processMultispectral(x):
    
    if config.processing_type == 'normalization':
        x = normalizeIMG(x)
    
    elif config.processing_type == 'standardization':
        mean = ms_eurosat_stats["mean"]
        std = ms_eurosat_stats["std"]
        x = standardizeIMG(x, mean, std)
    
    return x


def preprocess_tiff(image):
    # Clipping
    clip = ms_eurosat_stats["clip"]
    image = clip_img(image, clip)
    # Scale to 0..255
    image = scale_per_channel(image, clip, map_max=255)
    # Now it's the same as Resisc
    return image


def clip_img(img, clip_max):
    clip_max = np.reshape(clip_max, [1, 1, len(clip_max)])
    img = np.clip(img, 0, clip_max)
    return img

def scale_per_channel(img, clip_max, map_max=255):
    scale = map_max / clip_max
    scale = np.reshape(scale, [1, 1, len(scale)])
    img = img * scale
    # for i in range(img.shape[-1]):
    #     img[..., i] = img[..., i] * scale[i]
    return img

def visualize_all_bands(raw, bands=[3,2,1]):
    seg = raw[...,bands]   
    return seg.astype(int)





def unprocessRGB(x):
    
    if config.processing_type == 'normalization':
        x = x.astype(np.float32) #* 255
    
    elif config.processing_type == 'standardization':
        x = unstandardizeIMG(x, resisc_stats["mean"], resisc_stats["std"])
        x = x.astype(np.uint8)
        
    return x



def unprocessMultispectral(x):
    
    if config.processing_type == 'normalization':
        pass
    
    elif config.processing_type == 'standardization':
        x = unstandardizeIMG(x, ms_eurosat_stats["mean"], ms_eurosat_stats["std"])
        
    return x
    
                




##################Multispectral################################
def subImgFromMs(img_ms, mode="rgb", channel_idxs = []):
    """
    creates a array image with multispectral image channels specified
    
    Parameters
    ----------
    img_ms : multispectral image
        
    mode : type of output image
        if mode == "rgb", output image will be a concatenation of the R,G and B channels 
        from the multispectral image
        
    channel_idxs : sequence, channel index list

    Returns
    -------
        a array image
        if mode== rgb, output imageshape = (H, W, 3)
        else output image shape = (H, W, N) with N = len(channel_idxs)
        
    """
    
    img_shape = img_ms.shape
    
    if mode == "rgb":
        i = 1
        a = img_ms[:, :, i:i+3]
        b = np.zeros(a.shape)
        b[:, :, 0] = a[:, :, 2]
        b[:, :, 1] = a[:, :, 1]
        b[:, :, 2] = a[:, :, 0]
        
        return b
    
    
    elif channel_idxs != []: 
       
        nbrC = len(channel_idxs)
        cShape = (img_shape[0], img_shape[1], nbrC)
        c = np.zeros(cShape)
        for i, idx  in zip(range(nbrC), channel_idxs):
            c[:, :, i] = img_ms[:, :, idx]
   
        return c




def batch_ms2rgb(ms_batch):
    
    batch_shape = ms_batch.shape
    
    new_batch = np.zeros((batch_shape[0], batch_shape[1], batch_shape[2], 3))
    for i in range(batch_shape[0]):
       #new_batch[i, :, :, :] =  subImgFromMs(ms_batch[i, :, :, :], mode="rgb")
       new_batch[i] =  subImgFromMs(ms_batch[i], mode="rgb")
       
    return new_batch



def MsImgShowable(ms_img):
    """
    Transforms multispectral image into RGB image normalized between 0 and 1, so it can be plotted

    Parameters: multispectral image

    Returns: rgb image
    """
    
    rgb_img = subImgFromMs(ms_img, mode="rgb")
    rgb_img = normalizeIMG(rgb_img )
    return rgb_img.astype(np.float32)



#%%%%%%%%%%%% np nd-array image trasnformation 


def adjust_contrast(img, contrast_factor):
    """
    adjust_contrast
    For each channel, this Op computes the mean of the image pixels in the channel and 
    then adjusts each component x of each pixel to (x - mean) * contrast_factor + mean.

    Parameters
    ----------
    img : TYPE np nd-array 
        DESCRIPTION. image

    contrast_factor TYPE float
    
    Returns
    -------
    img_r

    """
    imgShape = img.shape
    img_r = np.zeros(imgShape)
    
    for i in range(imgShape[-1]):
        cmean = img[:, :, i].mean()
        img_r[:, :, i] =  (img[:, :, i] - cmean) * contrast_factor + cmean

    return img_r





def adjust_brightness(img, delta):
    """
    adjust_brightness
    on each channel, the value of scaled delta is added to all components of the tensor image
    scaled delta = mean(channel) * delta

    Parameters
    ----------
    image : TYPE np nd array
        DESCRIPTION.
    delta : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    imgShape = img.shape
    img_r = np.zeros(imgShape)
    
    for i in range(imgShape[-1]):
        cmean = img[:, :, i].mean()
        img_r[:, :, i] =  img[:, :, i] + cmean *  delta

    return img_r




def add_noise(img, mean = 0.0, std = 1.0):
    """
    add noise to np nd array image

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    mean : TYPE, optional
        DESCRIPTION. The default is 0.0.
    std : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    img_r : TYPE
        DESCRIPTION.

    """
    
     # some constant
     # some constant (standard deviation)
    #cmean = img.mean()
    img_r = img + np.random.normal(mean, std, img.shape)
    #noisy_img_clipped = np.clip(noisy_img, 0, 255) 

    return  img_r







def blur_img(img, sigma = (2, 2, 0)):
    """
    blur a np N-D image array

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    sigma : TYPE, optional sigma of gaussian_filter kernel  for each axis
        DESCRIPTION. The default is (2, 2, 0) for 3-D image 

    Returns
    -------
    s_blur : TYPE np N-D image array
        DESCRIPTION. blurred image

    """
    
    s_blur = gaussian_filter(img, sigma=sigma )
    
    return s_blur











