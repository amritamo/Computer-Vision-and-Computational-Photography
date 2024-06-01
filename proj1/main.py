# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import downscale_local_mean
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage import filters

import cv2

# name of the input file
imname = 'tobolsk.jpg'

# FOR IMAGE POST-PROCESSING
# imname = 'yurt_ip.jpg'

# read in the image (needed for auto-contrast function)
# im = cv2.imread(imname)


# read in the image
im = skio.imread(imname)



# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)


# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(int)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]



# align the images

#blue is the base image
# take red and green as template [-15, 15] pixel matrix and compute image optimization over blue image

#offset window size
pixel_offset = 15

#shifts the image by window size in x and y directions starting at pixel position x, y
def offset_img(img, x, y):
    x_shift = np.roll(img, x, axis = 1)
    y_shift = np.roll(x_shift, y, axis = 0)
    return y_shift

#compute and returns the best offset using ssd optimization
def best_offset_ssd(template_img, base_img, x_pixel, y_pixel):
    min_ssd = np.inf
    offset = [0,0]
    for i in range(x_pixel - pixel_offset, x_pixel + pixel_offset + 1):
        for j in range(y_pixel - pixel_offset , y_pixel + pixel_offset + 1):
            curr_ssd = np.sum((offset_img(template_img, i, j) - base_img) ** 2)
            if curr_ssd < min_ssd:
                min_ssd = curr_ssd
                offset[0] = i
                offset[1] = j
    print(offset)
    return offset

#compute and returns the best offset using ncc optimization
def best_offset_ncc(template_img, base_img, x_pixel, y_pixel):
    max_corr = 0
    offset = [0,0]
    for i in range(x_pixel - pixel_offset, x_pixel + pixel_offset + 1):
        for j in range(y_pixel - pixel_offset , y_pixel + pixel_offset + 1):
            product = np.mean((offset_img(template_img, i, j) - offset_img(template_img, i, j).mean()) * (base_img - base_img.mean()))
            stdev = offset_img(template_img, i, j).std() * base_img.std()
            if stdev == 0:
                offset = [i,j]
            else:
                curr_corr = product/stdev
                if max_corr < curr_corr:
                    max_corr = curr_corr
                    offset = [i,j]
    print(offset)
    return offset

# gets middle region of the image to avoid extra border around images after alignment. scale parameter lets user choose what percent of the image you want to keep. 
def get_center(img, scale):
    height = np.floor(img.shape[0]).astype(int)
    width = np.floor(img.shape[1]).astype(int)

    crop_height = np.floor(img.shape[0] * scale).astype(int)
    crop_width = np.floor(img.shape[1] * scale).astype(int)
   
    mid_x, mid_y = int(width/2), int(height/2)
    mid_cropw, mid_croph = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-mid_croph:mid_y+mid_croph, mid_x-mid_cropw:mid_x+mid_cropw]
    return crop_img

# Image alignment optimization for larger .tif images. Recursively calls itself on half the image size and keeps updating the offset based on the previous layer of the pyramid.
def image_pyramid(template_img, base_img):
    if template_img.shape[1] < 400:
        return best_offset_ssd(template_img, base_img, 0, 0)
    else:
        template_img_scaled = sk.transform.rescale(template_img, 0.5, anti_aliasing = True)
        base_img_scaled = sk.transform.rescale(base_img, 0.5,  anti_aliasing = True)
        offset = np.multiply(2, image_pyramid(template_img_scaled, base_img_scaled))
        return best_offset_ssd(template_img, base_img, offset[0], offset[1])

# gets center 80% of image
b_center = get_center(b, 0.8)
r_center = get_center(r, 0.8)
g_center = get_center(g, 0.8)


#ssd optimization
ar = offset_img(r_center, best_offset_ssd(r_center, b_center, 0, 0)[0], best_offset_ssd(r_center, b_center, 0, 0)[1])
ag = offset_img(g_center, best_offset_ssd(g_center, b_center, 0, 0)[0], best_offset_ssd(g_center, b_center, 0, 0)[1])

#ncc optimization
# ar = offset_img(r_center, best_offset_ncc(r_center, b_center, 0, 0)[0], best_offset_ncc(r_center, b_center, 0, 0)[1])
# ag = offset_img(g_center, best_offset_ncc(g_center, b_center, 0, 0)[0], best_offset_ncc(g_center, b_center, 0, 0)[1])

#image pyramid optimization
# print("red offset")
# ar = offset_img(r_center, image_pyramid(r_center, b_center)[0], image_pyramid(r_center, b_center)[1])
# print("green offset")
# ag = offset_img(g_center, image_pyramid(g_center, b_center)[0], image_pyramid(g_center, b_center)[1])


#BELLS AND WHISTLES

# Converts RGB to HSV space and equalizes V channel using .equalizeHist() function.
def auto_contrast(img):
    hsv_im = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_im[:,:,2] = cv2.equalizeHist(hsv_im[:,:,2])
    hist_eq_img = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
    return hist_eq_img

# im_out = auto_contrast(im)

# Maps pixel values to a different range such that the brightest pixel value is mapped to gray.
def white_balance(img):
    gs_img = rgb2gray(img)
    gs_img = np.multiply(gs_img, 255)
    brightest_pixel = 0
    for i in range(gs_img.shape[0]):
        for j in range(gs_img.shape[1]):
                if (gs_img[i][j] > brightest_pixel):
                    brightest_pixel = gs_img[i][j]
    scale = 220/brightest_pixel
    scale = scale.astype(int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for channel in range(3):
                img[i][j][channel] += scale
    return img

# im_out = white_balance(im)

# create a color image
im_out = np.dstack([ar, ag, b_center])

#convert image to uint8 dtype
im_out = img_as_ubyte(im_out)

#image with default alignment
# im_out = np.dstack([r_center, g_center, b_center])

# save the image
fname = '/Users/amrita/cs194-26/proj1/tobolsk_ssd.jpg'
skio.imsave(fname, im_out)


# display the image
skio.imshow(im_out)
skio.show()

# uncomment for auto-contrast function
# cv2.imwrite('yurt_hist_eq.jpg',im_out)
# cv2.imshow(' ', im_out)