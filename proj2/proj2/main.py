import matplotlib.pyplot as plt
from align_image_code import align_images
import scipy.signal as signal
import cv2
import skimage as sk
import skimage.io as skio
from skimage.color import rgb2gray
import numpy as np
from skimage.util import img_as_ubyte
import scipy.fft as fft

# # First load images

# # high sf
im1 = skio.imread('bean.jpg')/255
# im1 = rgb2gray(im1)
# im1=im1[:, :, None]

# # low sf
im2 = skio.imread('gw.jpg')/255
# im2 = rgb2gray(im2)
# im2=im2[:, :, None] 

# # Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)



## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

#Code I used to achieve image sharpening
def unsharp_mask_filter(img, ksize, sigma):
    img_blur = split_channels_apply_gaus(img, ksize, sigma)
    high_freq_img = img - img_blur
    return img + np.multiply(high_freq_img, 5)

#Creates a gaussian filter with specified kernel size and sigma
def get_gaussian_kernel_2d(ksize, sigma):
    k1d = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    return k1d * k1d.T

#Applies gaussian filter to color images
def split_channels_apply_gaus(img, ksize, sigma):
    (R, G, B) = cv2.split(img)
    gaussian_kernel = get_gaussian_kernel_2d(ksize, sigma)
    img_blur_b = signal.convolve2d(B, gaussian_kernel, mode = "same")
    img_blur_g = signal.convolve2d(G, gaussian_kernel, mode = "same")
    img_blur_r = signal.convolve2d(R, gaussian_kernel, mode = "same")
    merged = cv2.merge([img_blur_r, img_blur_g, img_blur_b])
    return merged

#Applies gaussian filter to grayscale images(Needed for alignment code)
def grayscale_apply_gaus(img, ksize, sigma):
    gaussian_kernel = get_gaussian_kernel_2d(ksize, sigma)
    img=img[:, :, 0] 
    img_blur = signal.convolve2d(img, gaussian_kernel, mode = "same")
    return img_blur

#Get high frequency of grayscale of RGB images
def get_high_freq(img, ksize, sigma):
    if img.shape[2] == 1:
        img_blur = grayscale_apply_gaus(img, ksize, sigma)
        high_freq_img = img[:,:,0] - img_blur
        high_freq_img = img - img_blur
    else:
        img_blur = split_channels_apply_gaus(img, ksize, sigma)
        high_freq_img = img - img_blur
    return high_freq_img

#Get low frequency of grayscale of RGB images
def get_low_freq(img, ksize, sigma):
    if img.shape[2] == 1:
        img_blur = grayscale_apply_gaus(img, ksize, sigma)
    else:
        img_blur = split_channels_apply_gaus(img, ksize, sigma)
    return img_blur




fname = '/Users/amrita/cs194-26/proj2/grayscale_gw.jpg'
skio.imsave(fname, im_out)
# # skio.imshow(im_out)
# # skio.show()