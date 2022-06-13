import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy.signal import convolve2d
from scipy import misc

import seaborn as sns
def show_image(imgs=[], names=[], option=[]):
    f,ax = plt.subplots(1,(len(imgs)),figsize=(10,10))
    #f.set_figheight(250)
    for i in range(len(imgs)):
        if option[i] is not None:
            ax[i].imshow(imgs[i], option[i])
        else:
            ax[i].imshow(imgs[i])
        ax[i].set_title(names[i])
        ax[i].axes.xaxis.set_visible(False)
        ax[i].axes.yaxis.set_visible(False)
    
def block_binary_filter(img ,block_size = 5, thresh_hold = 255/2):
    mask = np.zeros(img.shape, dtype=np.uint8)
    for x in range(int(img.shape[1]/block_size)):
        for y in range(int(img.shape[0]/block_size)):
            block =img[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size]
            #print(block)
            if np.average(block)<thresh_hold:
                mask[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size] = 0
            else:
                #@print(12)
                mask[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size] = 255
            #print(bin[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size])
    
    return mask
def img_filter(img):
    q = [4.0, 12.0, 2.0]
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]
    
    filters = filter1+filter2+filter3
    #print(filters)

    result = cv2.filter2D(img.copy(), -1, kernel=filters, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    return result

def generate_vector(img):
    
    kernel = np.zeros((3,3), dtype=float)
    kernel[1][0] = -0.5
    kernel[1][2] = 0.5
    
    y_vector = convolve2d(img, kernel, boundary='symm', mode='same')
    
    #y_vector = convolve2d(img.copy(), -1, kernel=kernel, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    sns.heatmap(y_vector,vmin=-50, vmax=50)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title('gradient y',fontsize=20)
    plt.show()
    plt.close()
    kernel = np.zeros((3,3), dtype=float)
    kernel[0][1] = -0.5
    kernel[2][1] = 0.5
    x_vector = convolve2d(img, kernel, boundary='symm', mode='same')
    #x_vector = convolve2d(img.copy(), -1, kernel=kernel, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    #print(np.min(x_vector))
    sns.heatmap(x_vector,vmin=-50, vmax=50)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title('gradient x',fontsize=20)
    plt.show()
    plt.close()
    vector_length = np.sqrt(x_vector**2+y_vector**2)
    return vector_length