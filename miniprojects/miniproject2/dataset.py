import numpy as np
import matplotlib.pyplot as plt

class xor(object):
    """ 
    Class that creates a xor dataset. Note that for the grading of the project, this method 
    might be changed, although it's output format will not be. This implies we might use other
    methods to create data. You must assume that the dataset will be blind and your machine is 
    capable of running any dataset. Although the dataset will not be changed drastically and will
    hold the XOR style . 
    """    
    def __init__(self, **kwargs):

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
        plt.pause(0.001)
        

class mystery(object):
    """ 
    Class that loads a mystery dataset. 
    """    
    def __init__(self, **kwargs):