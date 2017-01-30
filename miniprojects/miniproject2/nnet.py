import numpy as np

def sigmoid ( a ):
    """
    This method activates pointwise all elements of the arrays. 

    Args:
        a: input array.

    Returns:
        np.ndarray: same size and shape as a.
    """
    return 1.0/(1.0 + np.exp(-a))

def rmse ( a,  b ): 
    """
    This function produces a point-wise root mean squared error error between ``a`` and ``b``
    
    Args:
        a: first input ndarray
        b: second input ndarray

    Returns: 
        numpy float: rmse error 

    Notes:
        The grade that you will get will depend on this output. The lower this value, the higher 
        your grade.
    """        
    return np.sqrt(np.mean((a - b) ** 2))

class xor(object):
    """ 
    Class that creates a xor dataset. Note that for the grading of the project, this method 
    might be changed, although it's output format will not be. This implies we might use other
    methods to create data. You must assume that the dataset will be blind and your machine is 
    capable of running any dataset.
    
    Args:
        mu: tuple, means of the gaussian with which we add noise (optional, default 0)
        sigma: variance of the gaussian with which we add noise (optional, default 0.1)
    """    
    def __init__(self, **kwargs):
        low = 200
        high = 400
        if 'num_layers' in kwargs.keys():
            self.num_layers = kwargs['num_layers']
        else:
            self.num_layers = 2

        if 'dimensions' in kwargs.keys():
            self.dimensions = kwargs['dimensions']
        else:
            self.dimensions = np.random.randint(low = low,high = high)
        if 'mu' in kwargs.keys():
            self.mu = kwargs['mu']
        else:
            self.mu = 0
        if 'sigma' in kwargs.keys():
            self.sigma = kwargs['sigma']
        else:
            self.sigma = 0.1

        self.w = []
        self.b = []
        for layer in xrange(self.num_layers):
            self.w.append ( np.random.rand(self.dimensions,1) )
            self.b.append ( np.random.rand(1) )

    def query_data(self, **kwargs):
        """
        Once initialized, this method will create more data.

        Args:
            samples: number of samples of data needed (optional, default randomly 10k - 50k)   
        Returns:
            tuple: data a tuple, ``(x,y)``
                ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
                data and data is spread along axis 1. 
                ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.                          
        """
        if 'samples' in kwargs.keys():
            samples = kwargs['samples']
        else:
            samples = np.random.randint(low = 1000, high = 5000)

        y_temp = np.random.normal(size = (samples,self.dimensions))    
        for layer in xrange(self.num_layers):
            y_temp = sigmoid(np.dot(y_temp,self.w[layer]) + np.random.normal(self.mu, self.sigma,
                                (samples,1)) + self.b[layer] )
        
        return (x,y)