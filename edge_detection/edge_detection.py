
from imgutils import *
from scipy.ndimage import convolve1d as imfilter
import sys
sys.path.append( '../core' )
from imgutils import *

img = imread('led.jpg')  # Load the image
img = rgb2gray(img)   # convert to grayscale
imshow(img,'original image')
# Create filters
differentials = np.asarray([-1,0,1])

dx = imfilter(input = img, weights = differentials, axis = 0)  # This filters straight.
dy = imfilter(input = img, weights = differentials, axis = 1)  # This filters transposed.

# Display the differentials.
imshow(dx,'dx')
imshow(dy,'dy')

edge_map_threshold = 10     # This is the threshold above which, we shall call a differential an
                            # edge.

dx_em = np.zeros(dx.shape)
dx_em[dx > edge_map_threshold] = 255
dy_em = np.zeros(dy.shape)
dy_em[dy > edge_map_threshold] = 255


imshow(dx_em,'dx_em')
imshow(dy_em,'dy_em')

## Save the iamges down.
imwrite(dx,'dx.jpg')
imwrite(dy,'dy.jpg')
imwrite(dx_em,'dx_em.jpg')
imwrite(dy_em,'dy_em.jpg')