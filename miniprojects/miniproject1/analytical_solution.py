#sample_submission.py
import numpy as np
from numpy.linalg import pinv as inverse

class regressor(object):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """
    def __init__(self, data):
        self.x, self.y = data
        # Solving using the analytical solution
        self.x = np.concatenate((np.ones((self.x.shape[0],1)), self.x), axis = 1)
        self.w = np.dot(inverse(np.dot(self.x.transpose,self.x)),np.dot(self.x.transpose,self.y))

        self.b = self.x[1,] # First row is just b 
        self.w = self.w[1:] # w is all but the first row.

    def get_params (self):
        """ 
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return a random numpy array for demonstration purposes.

        """
        return (self.w, self.b)

    def get_predictions (self, x):
        """
        Method should return the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:    
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of 
                            ``x`` 
        Notes:
            Temporarily returns random numpy array for demonstration purposes.                            
        """        
        # Here is where you write a code to evaluate the data and produce predictions.
        return np.random.rand(self.x.shape[1])

if __name__ == '__main__':
    pass 
