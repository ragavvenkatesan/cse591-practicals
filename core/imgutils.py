from PIL import Image as I
import numpy as np

from skimage.util import random_noise as imnoise
from scipy.misc import imresize 

def imread( infile ) :
    """
    Takes a string as input and outputs a image.

    Args:
        infile: input file string
    
    Returns:
        numpy.ndarray: image 
    """
    img = I.open( infile)
    img.load()
    data = np.asarray( img, dtype="uint8" )
    return data

def imwrite( img, outfile ): 
    """
    Takes an image and string as input and and writes the image down.

    Args:
        img: image numpy.ndarray
        outfile: output file string
    """    
    if not img.dtype == 'uint8':
        np.clip(img, a_min = 0, a_max = 255, out = img)
        img = np.asarray(img, dtype = 'uint8')
    img = I.fromarray( img )
    img.save( outfile )

def imshow ( img , window = 'image'):
    """
    Takes an image as input and displays it.

    Args:
        img: image numpy.ndarray
        window: (optional) title to name the window by.
    """        
    I.fromarray(img).show(title = window)

def rgb2gray(rgb):
    """
	Function that takes as input one rgb image array and returns a grayscale image. It applies 
	the following transform:

	.. math::
	
		I_{gray} = 0.2989I_r + 0.5870I_g + 0.1140I_b

	Args:
		rgb: ``numpy ndarray`` of a four-dimensional image batch of the form 
									< (number of images??), height, width, channels>

	Returns:
		numpy ndarray: gray
	"""
    if len(rgb.shape) == 4:
        r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    elif len(rgb.shape) == 3:
        r, g, b = rgb[:,:,0], rgb [:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

if __name__ == '__main__':
    pass