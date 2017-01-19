"""
Detect edges. 
=============

Use this code to run some basic edge detection code on some images of your own choosing. 
Have fun!

Setup.
------

Use pip insallers to get these packages.

    pip install pillow
    pip install numpy
    pip install scipy

For those who want to use anaconda distributions of python, use

    conda install numpy
    conda install scipy
"""
from imgutils import *
from scipy.ndimage import convolve1d as imfilter

img = imread('led.jpg')  # Load the image
img = rgb2gray(img)   # convert to grayscale
imshow(img,'original image')
# Create filters
differentials = np.asarray([-1,0,1])

dx = imfilter(input = img, weights = differentials, axis = 0)
dy = imfilter(input = img, weights = differentials, axis = 1)

imshow(dx,'dx')
imshow(dy,'dy')

dx_em = np.zeros(dx.shape)
dx_em[dx > 10] = 255
dy_em = np.zeros(dy.shape)
dy_em[dy > 10] = 255


imshow(dx_em,'dx_em')
imshow(dy_em,'dy_em')

## Save the iamges down.
imwrite(dx,'dx.jpg')
imwrite(dy,'dy.jpg')
imwrite(dx_em,'dx_em.jpg')
imwrite(dy_em,'dy_em.jpg')