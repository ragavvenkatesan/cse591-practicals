# will ``sample_submission`` to your submission filename.

from sample_submission import regressor
import numpy as np

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

class dataset_generator(object):
    """ 
    Class that creates a random dataset. Note that for the grading of the project, this method 
    might be changed, although it's output format will not be. This implies we might use other
    methods to create data. You must assume that the dataset will be blind and your machine is 
    capable of running any dataset.
    
    Args:
        dimensions: number of dimensions of dataset (optional, default randomly 15-30)
        mu: mean of the gaussian with which we add noise (optional, default 0)
        sigma: variance of the gaussian with which we add noise (optional, default 0.1)
    """    
    def __init__(self, **kwargs):
        low = 15
        high = 30
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

        self.w = np.random.rand(self.dimensions,1)
        self.b = np.random.rand(1)

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

        x = np.random.uniform(size = (samples,self.dimensions))        
        y = np.dot(x,self.w) + np.random.normal(self.mu, self.sigma, (samples,1)) + self.b
        
        return (x,y)

if __name__ == '__main__':
    
    dg = dataset_generator() # Initialize a dataset creator
    data_train = dg.query_data(samples = 5000) # Create a random training dataset.

    r = regressor(data_train)  # This call should return a regressor object that is fully trained.
    params = r.get_params()    # This call should reaturn parameters of the model that are 
                               # fully trained.

    data_test = dg.query_data(samples = 5000)  # Create a random testing dataset.
    predictions = r.get_predictions(data_test[0]) # This call should return predictions.

    print "Rmse error of predictions = " + str(rmse(data_test[1], predictions))