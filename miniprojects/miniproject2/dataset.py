import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append( '../../core' )
from imgutils import *

class xor(object):
    """ 
    Class that creates a xor dataset. Note that for the grading of the project, this method 
    might be changed, although it's output format will not be. This implies we might use other
    methods to create data. You must assume that the dataset will be blind and your machine is 
    capable of running any dataset. Although the dataset will not be changed drastically and will
    hold the XOR style . 
    """    
    def __init__(self):

        self.dimensions = 2
        self.positive_means = [[-1,-1],[1,1]]
        self.negative_means = [[-1,1],[1,-1]]
        self.covariance = [[0.35, 0.1], [0.1, 0.35]]

    def query_data(self, **kwargs):
        """
        Once initialized, this method will create more data.

        Args:
            samples: number of samples of data needed (optional, default randomly 10k - 50k)   
        Returns:
            tuple: data a tuple, ``(x,y)``
                ``x`` is a two dimensional ndarray ordered such that axis 0 is independent 
                data and data is spread along axis 1. 
                ``y`` is a 1D ndarray it will be of the same length as axis 0 of x.                          
        """
        if 'samples' in kwargs.keys():
            samples = kwargs['samples']
        else:
            samples = np.random.randint(low = 1000, high = 5000)

        # make positive samples
        dim1, dim2 = np.random.multivariate_normal( self.positive_means[0], 
                                                    self.covariance, samples/4).T
        positive = np.stack((dim1,dim2),axis = 1)
        dim1, dim2 = np.random.multivariate_normal( self.positive_means[1], 
                                                    self.covariance, samples/4).T            
        positive = np.concatenate((positive,np.stack((dim1,dim2),axis = 1)),axis = 0)
        labels = np.ones(positive.shape[0])

        # make the negative samples
        dim1, dim2 = np.random.multivariate_normal( self.negative_means[0], 
                                                    self.covariance, samples/4).T
        negative = np.stack((dim1,dim2),axis = 1)
        dim1, dim2 = np.random.multivariate_normal( self.negative_means[1], 
                                                    self.covariance, samples/4).T            
        negative = np.concatenate((negative,np.stack((dim1,dim2),axis = 1)),axis = 0)    
        labels = np.concatenate((labels,np.zeros(negative.shape[0])), axis = 0)

        data = np.concatenate((positive, negative),axis = 0)        
        assert data.shape[0] == labels.shape[0]
  
        perm = np.random.permutation(labels.shape[0])
        data = data[perm,:]
        labels = labels[perm]

        return (data,np.asarray(labels,dtype = 'int'))

    def plot(self, data,labels):
        """
        This method will plot the data as created by this dataset generator.

        Args:
            data: as produced by the ``query_data`` method's first element.
            labels: as produced by the ``query_data`` method's second element.
        """
        positive = data[labels == 1,:]
        negative = data[labels == 0,:]

        plt.plot(positive[:,0], positive[:,1], 'bo', negative[:,0], negative[:,1], 'rs')        
        plt.axis('equal')      
        plt.title('XOR Dataset')  
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()

    def _demo (self):
        """
        This is a demonstration method that will plot a version of the dataset on the screen.
        """        
        data, labels = self.query_data(samples = 5000) 
        self.plot(data, labels)        
        

class waldo(object):
    """ 
    Class that creates the waldo dataset. 

    Args:
        dimensions: <tuple> the dimensions of image (optional, default randomly (28,28)) 
        noise: controls the variance of the noise being applied.    
        img: load and use an image that is not default ('waldo.jpg')
    """    
    def __init__(self, **kwargs):
        if 'dimensions' in kwargs.keys():
            self.sample_height = kwargs['dimensions'][0]
            self.sample_width = kwargs['dimensions'][1]
        else:
            self.sample_height = 28
            self.sample_width = 28
            
        if 'img' in kwargs.keys():
            img = kwargs['img']
        else:
            img = 'waldo.jpg'            

        if 'noise' in kwargs.keys():
            self.var = kwargs['noise']
        else:
            if self.sample_width < 32 and self.sample_height < 32:
                self.var = 0.05
            elif self.sample_width < 64 and self.sample_heigh < 64:
                self.var = 0.2
            else:
                self.var = 0.7

        img = imread(img)  # Load the image
        self.waldo = rgb2gray(img)   # convert to grayscale  
        self.waldo = normalize(self.waldo)      
        self.reshape_low_height = np.floor(self.sample_height * 0.35)  
        self.reshape_high_height = np.floor(self.sample_height * 0.95)
        self.reshape_low_width = np.floor(self.sample_width * 0.35)  
        self.reshape_high_width = np.floor(self.sample_width * 0.95)

    def _query_positive_sample (self):
        """
        This is an internal method that creates positive data samples. 

        Notes:
            This creates one sample. 
        """     
        sample = self._query_negative_sample().reshape(self.sample_height, self.sample_width)      
        rshp = (np.random.randint (low = self.reshape_low_height, high =self.reshape_high_height + 1),
                np.random.randint (low = self.reshape_low_width, high = self.reshape_high_width + 1))
        waldo_reshaped = imresize(self.waldo, size = rshp)
        waldo_sample = imnoise(waldo_reshaped, mode = 'gaussian', var = self.var, clip = True)  
        waldo_sample = imnoise(waldo_sample, mode = 's&p', clip = True) * 255
        current_waldo_height = waldo_sample.shape[0]
        current_waldo_width = waldo_sample.shape[1] 
        height_low = 1
        height_high = self.sample_height - current_waldo_height - 1
        width_low = 1
        width_high = self.sample_width - current_waldo_width - 1
        waldo_x_pos = np.random.randint(low = height_low, high = height_high + 1)
        waldo_y_pos = np.random.randint(low = width_low, high = width_high + 1)
        sample[ waldo_x_pos : waldo_x_pos + current_waldo_height,
                waldo_y_pos : waldo_y_pos + current_waldo_width ] = 0.7 * waldo_sample + \
                0.3 * sample[ waldo_x_pos : waldo_x_pos + current_waldo_height,
                        waldo_y_pos : waldo_y_pos + current_waldo_width ]               
        return np.asarray(sample,dtype = 'uint8').flatten()

    def _query_negative_sample (self):
        """
        This is an internal method that creates negative data samples. 

        Notes:
            This creates one sample. 
        """             
        sample = np.random.randint(low = 0, high = 256, 
                                    size = (self.sample_height, self.sample_width))
        return np.asarray(sample,dtype = 'uint8').flatten()

    def query_data (self, **kwargs):
        """
        Once initialized, this method will create data.

        Args:
            samples: number of samples of data needed (optional, default randomly 10k - 50k)   
        Returns:
            tuple: data a tuple, ``(x,y)``
                ``x`` is a two dimensional ndarray ordered such that axis 0 is independent 
                data and data is spread along axis 1. 
                ``y`` is a 1D ndarray it will be of the same length as axis 0 or x. Will be integer.                        
        """
        if 'samples' in kwargs.keys():
            samples = kwargs['samples']
        else:
            samples = np.random.randint(low = 100, high = 500)  

        # Create dummy arrays
        data = np.zeros((samples,self.sample_height*self.sample_width))
        labels = np.zeros((samples,), dtype ='int')

        for sample in xrange(samples): 
            labels[sample] = np.random.randint(low = 0, high = 2)
            if labels[sample] == 1:
                data[sample] = self._query_positive_sample()
            else:
                data[sample] = self._query_negative_sample()
        
        return (data,labels)
            
    def _demo(self):
        """
        This is a demonstration method that will display a random positive and negative samples.
        """
        sample_positive = self._query_positive_sample().reshape(self.sample_height, 
                                                                        self.sample_width)
        imshow(sample_positive,window = 'positive')

        sample_negative = self._query_negative_sample().reshape(self.sample_height, 
                                                                        self.sample_width)
        imshow(sample_negative,window = 'negative')

    def display_sample (self, sample, title = 'image'):
        """
        This method will display a particular smaple in the dataset generated.

        Args:
            sample: provide one row of data
            title: (optional) title of the window for image display
        """
        imshow(sample.reshape(self.sample_height, self.sample_width), window = title)
