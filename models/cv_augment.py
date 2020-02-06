# From https://github.com/jbohnslav/opencv_transforms/blob/master/opencv_transforms/functional.py
import cv2
import numpy as np


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    table = np.array([ i*brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img, table)


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an mage.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    table = np.array([ (i-74)*contrast_factor+74 for i in range (0,256)]).clip(0,255).astype('uint8')
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img,table)
