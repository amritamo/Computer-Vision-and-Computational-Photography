#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import cv2
import scipy.signal as signal
import scipy.fft as fft

import numpy as np
import skimage.io as skio

import math
import skimage.transform as sktr
import skimage as sk
import skimage.io as skio

from skimage.color import rgb2gray


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

#applies gaussian filter to blur images
def get_gaussian_kernel_2d(ksize, sigma):
    k1d = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    return k1d * k1d.T


# ## Part 1: Fun with Filters

# In[5]:


#Convert images to grayscale 
def rgb2gray(rgb):
    return rgb.dot([0.2989, 0.5870, 0.1140])


# In[6]:


img = skio.imread("cameraman.jpg")
img_gray = rgb2gray(img);
print(img_gray.shape)
plt.figure(figsize=(12, 4));
plt.subplot(122); plt.imshow(img_gray, cmap="gray");


# In[8]:


#Blur camera man image
gaussian_kernel = get_gaussian_kernel_2d(11, 3)
img_blur = signal.convolve2d(img_gray, gaussian_kernel, mode="same");
plt.imshow(img_blur, cmap="gray"); 


# ### Finite Difference Operator

# In[9]:


def get_dx_op():
    dx_op = np.tile(np.array([1, -1])[None,:], [2, 1]) /2
    return dx_op

def get_dy_op():
    dy_op = np.tile(np.array([1, -1])[:, None], [1, 2]) /2
    return dy_op


# In[10]:


dx_op = get_dx_op()
print(dx_op)
plt.imshow(dx_op, cmap = "gray");


# In[11]:


dy_op = get_dy_op()
print(dy_op.shape)
print(dy_op)
plt.imshow(dy_op, cmap = "gray");


# In[12]:


#convolved with dx
img_partial_dx = signal.convolve2d(img_gray, dx_op, mode="same");
plt.imshow(img_partial_dx, cmap="gray"); 


# In[13]:


#convolve blurred image with Dx
img_blur_dx = signal.convolve2d(img_blur, dx_op, mode="same");
plt.imshow(img_blur_dx, cmap="gray"); 


# In[11]:


#convolved with dy
img_partial_dy = signal.convolve2d(img_gray, dy_op, mode="same");
plt.imshow(img_partial_dy, cmap="gray"); 


# In[14]:


#convolve blurred image with Dy

img_blur_dy = signal.convolve2d(img_blur, dy_op, mode="same");
plt.imshow(img_blur_dy, cmap="gray"); 


# In[15]:


#get gradient magnitude of image
gradient_magnitude = (((img_partial_dx)**2 + (img_partial_dy)**2)**0.5)
plt.imshow(gradient_magnitude, cmap="gray");


# In[16]:


#get gradient magnitude of blurred image

blurred_gradient_magnitude = (((img_blur_dx)**2 + (img_blur_dy)**2)**0.5)
plt.imshow(blurred_gradient_magnitude, cmap="gray");


# In[17]:


#create edge image for image

thresh = 55

maxval = 255

edge_image = (gradient_magnitude > thresh) * maxval
plt.imshow(edge_image, cmap="gray");


# In[18]:


#create edge image for blurred image

blur_thresh = 7

blur_edge_image = (blurred_gradient_magnitude > blur_thresh) * maxval
plt.imshow(edge_image, cmap="gray");


# ### Derivative of gaussians

# We get the derivative of Gaussian filter by convolving the Gaussian filter with the finite difference operator

# In[17]:


gaussian_kern = get_gaussian_kernel_2d(11, 2)
gauss_kernel_dx = signal.convolve2d(gaussian_kern, dx_op, mode="same")
plt.imshow(gauss_kernel_dx, cmap = "gray");


# In[18]:


gauss_kernel_dy = signal.convolve2d(gaussian_kern, dy_op, mode="same")
plt.imshow(gauss_kernel_dy, cmap = "gray");


# In[19]:


single_conv_dx = signal.convolve2d(img_gray, gauss_kernel_dx, mode="same");
plt.imshow(single_conv_dx, cmap="gray"); 


# In[20]:


single_conv_dy = signal.convolve2d(img_gray, gauss_kernel_dy, mode="same");
plt.imshow(single_conv_dy, cmap="gray"); 


# In[21]:


single_conv_gradient_magnitude = (((single_conv_dx)**2 + (single_conv_dy)**2)**0.5)
plt.imshow(single_conv_gradient_magnitude, cmap="gray");


# In[22]:


single_conv_thresh = 7

single_conv_edge_image = (single_conv_gradient_magnitude > single_conv_thresh) * maxval
plt.imshow(single_conv_edge_image, cmap="gray");


# # Part 2.3/2.4

# In[23]:


from skimage.color import rgb2gray

im_a = skio.imread("apple.jpeg")/255


im_b = skio.imread("orange.jpeg")/255

plt.subplot(121); plt.imshow(im_a);
plt.subplot(122); plt.imshow(im_b);


# In[24]:


#Apply gaussian blur for color images
def split_channels_apply_gaus(img, ksize, sigma):
    (R, G, B) = cv2.split(img)
    gaussian_kernel = get_gaussian_kernel_2d(ksize, sigma)
    img_blur_b = signal.convolve2d(B, gaussian_kernel, mode = "same")
    img_blur_g = signal.convolve2d(G, gaussian_kernel, mode = "same")
    img_blur_r = signal.convolve2d(R, gaussian_kernel, mode = "same")
    merged = cv2.merge([img_blur_r, img_blur_g, img_blur_b])
    return merged

#Apply gaussian blur for grayscale images
def grayscale_apply_gaus(img, ksize, sigma):
    gaussian_kernel = get_gaussian_kernel_2d(ksize, sigma)
    img_blur = signal.convolve2d(img, gaussian_kernel, mode = "same")
    return img_blur

#Get high frequencies for color or grayscale images
def get_high_freq(img, ksize, sigma):
    if len(img.shape) == 2:
        img_blur = grayscale_apply_gaus(img, ksize, sigma)
        high_freq_img = img - img_blur
    else:
        img_blur = split_channels_apply_gaus(img, ksize, sigma)
        high_freq_img = img - img_blur
    return high_freq_img

#Get low frequencies for color or grayscale images
def get_low_freq(img, ksize, sigma):
    if len(img.shape) == 2:
        img_blur = grayscale_apply_gaus(img, ksize, sigma)
    else:
        img_blur = split_channels_apply_gaus(img, ksize, sigma)
    return img_blur


# In[25]:


#Create gaussian stack
def gaussian_stack(img, kernel_size, sigma):
    gs = [img]
    for i in range(1, 5):
        im = get_low_freq(img, kernel_size, sigma)
        gs.append(im)
        img = im
        i = i+1
    return gs
    

#Create laplacian stack
def laplacian_stack(gs):
    ls = []
    for i in range(5):
        if i == 4:
            ls.append(gs[4])
        else:
            ls.append(gs[i] - gs[i+1])
    return ls


# In[26]:


#Create gaussian stacks for apple and orange
gaus_a = gaussian_stack(im_a, 6, 1)
gaus_b = gaussian_stack(im_b, 6, 1)


# In[27]:


plt.subplot(121); plt.imshow(gaus_a[0]);
plt.subplot(122); plt.imshow(gaus_a[1]);

# plt.subplot(121); plt.imshow(gaus_b[0]);
# plt.subplot(122); plt.imshow(gaus_b[4]);


# In[28]:


#output laplacian stacks for apple and orange
ls_a = laplacian_stack(gaus_a)
a0 = ls_a[2]
ls_a[2] = (a0 - np.min(a0)) / (np.max(a0) - np.min(a0))

plt.imshow(ls_a[2])

ls_b = laplacian_stack(gaus_b)
b0 = ls_b[2]
# print(b0)
ls_b[2] = (b0 - np.min(b0)) / (np.max(b0) - np.min(b0))

plt.imshow(ls_b[2])


# In[29]:


#Add all levels of laplacian stack to make sure it is the original
sum0 = np.add(ls_b[0], ls_b[1])
sum0 = (sum0 - np.min(sum0)) / (np.max(sum0) - np.min(sum0))

sum1 = np.add(sum0, ls_b[2])
sum1 = (sum1 - np.min(sum1)) / (np.max(sum1) - np.min(sum1))

sum2 = np.add(sum1, ls_b[3])
sum2 = (sum2 - np.min(sum2)) / (np.max(sum2) - np.min(sum2))
sum3 = np.add(sum2, ls_b[4])
sum3 = (sum3 - np.min(sum3)) / (np.max(sum3) - np.min(sum3))
plt.imshow(sum3)


# In[30]:


#create binary mask
mask = im_a.copy()
(h, w) = im_a.shape[:2]

for i in range(h):
    for j in range(w):
        if j <= h/2:
            mask[i][j] = 1
        else:
            mask[i][j] = 0
plt.imshow(mask)

# skio.imsave('/home/jovyan/Untitled Folder/binary_mask.jpg', mask)


# In[31]:


#create gaussian stack for mask
gaus_mask = gaussian_stack(mask, 20, 10)
im_gaus = gaus_mask[1]
im_gaus = (im_gaus - np.min(im_gaus)) / (np.max(im_gaus) - np.min(im_gaus))
plt.imshow(im_gaus)
# skio.imsave('/home/jovyan/Untitled Folder/binary_mask_blur.jpg', im_gaus)


# In[32]:


#applies mask for color images
def split_channels_apply_mask(img, mask):
    (aR, aG, aB) = cv2.split(img)
    (mR, mG, mB) = cv2.split(mask)
        
    combined_R = apply_mask(aR, mR)
    combined_G = apply_mask(aG, mG)
    combined_B = apply_mask(aB, mB)
    merged = cv2.merge([combined_R, combined_G, combined_B])
    return merged


# In[33]:


def apply_mask(img, mask):
    (h, w) = img.shape[:2]
    ls = img
    for i in range(h):
        for j in range(w):
            g = mask[i][j]
            a = ls[i][j]
            #ls[i][j] = ((g) * a) for apple
            ls[i][j] = ((1-g) * a)
    return ls


# In[34]:


# masked_apple = split_channels_apply_mask(ls_a[2], gaus_mask[2])
# masked_apple = (masked_apple - np.min(masked_apple)) / (np.max(masked_apple) - np.min(masked_apple))

# plt.imshow(masked_apple)


masked_orange = split_channels_apply_mask(ls_b[4], gaus_mask[4])
masked_orange = (masked_orange - np.min(masked_orange)) / (np.max(masked_orange) - np.min(masked_orange))

plt.imshow(masked_orange)


# In[35]:


#applies combined_laplacian for a color image
def split_channels_combine_laplacian(a, b, mask):
    (aR, aG, aB) = cv2.split(a)
    (bR, bG, bB) = cv2.split(b)
    (mR, mG, mB) = cv2.split(mask)
        
    combined_R = combined_laplacian(aR, bR, mR)
    combined_G = combined_laplacian(aG, bG, mG)
    combined_B = combined_laplacian(aB, bB, mB)
    merged = cv2.merge([combined_R, combined_G, combined_B])
    return merged


# In[36]:


#combines apple and orange image for each layer of laplacian
def combined_laplacian(la, lb, gr_mask):
    (h, w) = la.shape[:2]
    cl = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            g = gr_mask[i][j]
            a = la[i][j]
            b = lb[i][j]
            cl[i][j] = ((g * a) + (1-g)*b)  

    return cl


# In[37]:


comb1 = split_channels_combine_laplacian(ls_a[0], ls_b[0], gaus_mask[0])
comb1 = (comb1 - np.min(comb1)) / (np.max(comb1) - np.min(comb1))

comb2 = split_channels_combine_laplacian(ls_a[1], ls_b[1], gaus_mask[1])
comb2 = (comb2 - np.min(comb2)) / (np.max(comb2) - np.min(comb2))

comb3 = split_channels_combine_laplacian(ls_a[2], ls_b[2], gaus_mask[2])
comb3 = (comb3 - np.min(comb3)) / (np.max(comb3) - np.min(comb3))

comb4 = split_channels_combine_laplacian(ls_a[3], ls_b[3], gaus_mask[3])
comb4 = (comb4 - np.min(comb4)) / (np.max(comb4) - np.min(comb4))

comb5 = split_channels_combine_laplacian(ls_a[4], ls_b[4], gaus_mask[4])
comb5 = (comb5 - np.min(comb5)) / (np.max(comb5) - np.min(comb5))
plt.imshow(comb5)


# In[38]:


#sum all combined levels of laplacian stack to get final oraple
sum0 = np.add(comb1, comb2)
sum0 = (sum0 - np.min(sum0)) / (np.max(sum0) - np.min(sum0))

sum1 = np.add(sum0, comb3)
sum1 = (sum1 - np.min(sum1)) / (np.max(sum1) - np.min(sum1))

sum2 = np.add(sum1, comb4)
sum2 = (sum2 - np.min(sum2)) / (np.max(sum2) - np.min(sum2))

sum3 = np.add(sum2, comb5)
sum3 = (sum3 - np.min(sum3)) / (np.max(sum3) - np.min(sum3))

plt.imshow(sum3)


# In[39]:


xxx = skio.imread("xxx.jpg")/255
xxx = cv2.resize(xxx, (300, 300));



queen = skio.imread("queen.jpg")/255
queen = queen[:1250, 250:1650]

queen = cv2.resize(queen, (300, 300));

plt.imshow(queen)


# In[40]:


plt.imshow(xxx)


# In[41]:


#create gaussian stacks for xxx and the queen
gaus_x = gaussian_stack(xxx, 6, 1)
gaus_q = gaussian_stack(queen, 6, 1)


# In[42]:


#create laplacian stacks for xxx and the queen

ls_x = laplacian_stack(gaus_x)
ls_q = laplacian_stack(gaus_q)


# In[43]:


#combine laplacian stacks with mask

comb1 = split_channels_combine_laplacian(ls_x[0], ls_q[0], gaus_mask[0])
comb1 = (comb1 - np.min(comb1)) / (np.max(comb1) - np.min(comb1))

comb2 = split_channels_combine_laplacian(ls_x[1], ls_q[1], gaus_mask[1])
comb2 = (comb2 - np.min(comb2)) / (np.max(comb2) - np.min(comb2))

comb3 = split_channels_combine_laplacian(ls_x[2], ls_q[2], gaus_mask[2])
comb3 = (comb3 - np.min(comb3)) / (np.max(comb3) - np.min(comb3))

comb4 = split_channels_combine_laplacian(ls_x[3], ls_q[3], gaus_mask[3])
comb4 = (comb4 - np.min(comb4)) / (np.max(comb4) - np.min(comb4))

comb5 = split_channels_combine_laplacian(ls_x[4], ls_q[4], gaus_mask[4])
comb5 = (comb5 - np.min(comb5)) / (np.max(comb5) - np.min(comb5))
plt.imshow(comb5)


# In[44]:


#add each level of combined laplacian stack to get final image
sum0 = np.add(comb1, comb2)
sum0 = (sum0 - np.min(sum0)) / (np.max(sum0) - np.min(sum0))

sum1 = np.add(sum0, comb3)
sum1 = (sum1 - np.min(sum1)) / (np.max(sum1) - np.min(sum1))

sum2 = np.add(sum1, comb4)
sum2 = (sum2 - np.min(sum2)) / (np.max(sum2) - np.min(sum2))

sum3 = np.add(sum2, comb5)
sum3 = (sum3 - np.min(sum3)) / (np.max(sum3) - np.min(sum3))

plt.imshow(sum3)


# In[45]:


teletubby_mask = skio.imread("cropped_teletubby_mask.jpg")/255
teletubby_mask = cv2.resize(teletubby_mask, (300, 200));
print(teletubby_mask.shape)


teletubby = skio.imread("cropped_teletubby.jpg")/255
teletubby = cv2.resize(teletubby, (300, 200));

plt.imshow(teletubby_mask)


# In[46]:


berk = skio.imread("berk.jpg")/255
berk = cv2.resize(berk, (300, 200));


print(berk.shape)
plt.imshow(berk)
skio.imsave('/home/jovyan/Untitled Folder/berk.jpg', berk)


# In[47]:


#create gaussian stacks for teletubby and berkeley

gaus_t = gaussian_stack(teletubby, 5, 0.25)
gaus_b = gaussian_stack(berk, 5, 0.25)


# In[48]:


#create laplacian stacks for teletubby and berkeley

ls_t = laplacian_stack(gaus_t)
ls_b = laplacian_stack(gaus_b)


# In[49]:


#create gaussian stack for mask
gaus_mask = gaussian_stack(teletubby_mask, 11, 3)


# In[50]:


comb1 = split_channels_combine_laplacian(ls_b[0], ls_t[0], gaus_mask[0])
comb1 = (comb1 - np.min(comb1)) / (np.max(comb1) - np.min(comb1))

comb2 = split_channels_combine_laplacian(ls_b[1], ls_t[1], gaus_mask[1])
comb2 = (comb2 - np.min(comb2)) / (np.max(comb2) - np.min(comb2))

comb3 = split_channels_combine_laplacian(ls_b[2], ls_t[2], gaus_mask[2])
comb3 = (comb3 - np.min(comb3)) / (np.max(comb3) - np.min(comb3))

comb4 = split_channels_combine_laplacian(ls_b[3], ls_t[3], gaus_mask[3])
comb4 = (comb4 - np.min(comb4)) / (np.max(comb4) - np.min(comb4))

comb5 = split_channels_combine_laplacian(ls_b[4], ls_t[4], gaus_mask[4])
comb5 = (comb5 - np.min(comb5)) / (np.max(comb5) - np.min(comb5))
plt.imshow(comb5)
# skio.imsave('/home/jovyan/Untitled Folder/comb_teletubby_berk.jpg', comb5)


# In[51]:


#add each level of combined laplacian stack to get final image

sum0 = np.add(comb1, comb2)
sum0 = (sum0 - np.min(sum0)) / (np.max(sum0) - np.min(sum0))

sum1 = np.add(sum0, comb3)
sum1 = (sum1 - np.min(sum1)) / (np.max(sum1) - np.min(sum1))

sum2 = np.add(sum1, comb4)
sum2 = (sum2 - np.min(sum2)) / (np.max(sum2) - np.min(sum2))

sum3 = np.add(sum2, comb5)
sum3 = (sum3 - np.min(sum3)) / (np.max(sum3) - np.min(sum3))

plt.imshow(sum3)


# ### FFT Plots from hybrid image section

# In[52]:


bean = skio.imread("bean.jpg")
bean = rgb2gray(bean)


# In[61]:


#fft of grayscale george washington
gw = skio.imread("gw.jpg")
gw = rgb2gray(gw)
gw_fft= np.log(np.abs(np.fft.fftshift(np.fft.fft2(gw))))
plt.imshow(gw_fft, cmap = "gray")


# In[62]:


#fft of grayscale mr.bean

bean_fft= np.log(np.abs(np.fft.fftshift(np.fft.fft2(bean))))
plt.imshow(bean_fft, cmap = "gray")
skio.imsave('/home/jovyan/Untitled Folder/bean_fft.jpg', bean_fft)


# In[63]:


#fft of grayscale high frequency mr.bean

high_freq_bean = skio.imread("high_freq_bean.jpg")
high_freq_bean_fft= np.log(np.abs(np.fft.fftshift(np.fft.fft2(high_freq_bean))))
plt.imshow(high_freq_bean_fft, cmap = "gray")


# In[64]:


#fft of grayscale low frequency george washington

low_freq_gw = skio.imread("low_freq_gw.jpg")
low_freq_gw_fft= np.log(np.abs(np.fft.fftshift(np.fft.fft2(low_freq_gw))))
plt.imshow(low_freq_gw_fft, cmap = "gray")


# In[59]:


hybrid = skio.imread("align_bean_gw.jpg")
plt.imshow(hybrid, cmap = "gray")


# In[67]:


#fft of grayscale hybrid image

hybrid_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid))))
plt.imshow(hybrid_fft, cmap = "gray")
skio.imsave('/home/jovyan/Untitled Folder/hybrid_fft.jpg', hybrid_fft)


# In[ ]:




