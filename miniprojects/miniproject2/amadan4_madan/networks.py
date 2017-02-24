"""MIT License

Copyright (c) 2017 Aarav Madan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np

class xor_net(object):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.

	labels: A 1D ndarray of the same length as axis 0 or x.  
                          
    """       
    def __init__(self, data, labels):

        self.x = data
        self.y = labels
        self.y = np.reshape(self.y, (self.y.shape[0],1)) 
        self.params = [] 
        self.dataset_size = self.x.shape[0] 
        self.threshold = 0.0000001       

        self.ita = 0.001 
        
        self.input_layer_nodes = self.x.shape[1]  + 1 
        self.hidden_layer_nodes = 10 
        self.outer_layer_nodes = 1  
        
        self.hidden_layer_weights = np.random.randn(self.input_layer_nodes, self.hidden_layer_nodes) - 1 
        self.outer_layer_weights = np.random.randn(self.hidden_layer_nodes, self.outer_layer_nodes) - 1 
        
        self.input_layer_output = np.concatenate((self.x, np.ones((self.dataset_size,1))), axis = 1)
 
        self.hidden_layer_input = np.zeros((self.dataset_size, self.hidden_layer_nodes)) 
        self.hidden_layer_output = np.zeros((self.dataset_size, self.hidden_layer_nodes))
                
        self.outer_layer_input = np.zeros((self.dataset_size, self.outer_layer_nodes)) 
        self.outer_layer_output = np.zeros((self.dataset_size, self.outer_layer_nodes))

        check_parent = isinstance(self, xor_net)
        check_child = isinstance(self, mlnn)

        if (check_parent and check_child) is False:
            self.train_model()
            
                   
    def cost(self):
        """ 
        Method that returns the error at any specified values of the output from outer layer and the given ground truth labels 

        Returns:
            float: mean squared error. 

        Notes:
            This code will return the mean of sum of sqaured errors at any given iteration during the gradient descent.

        """
        return 0.5*np.sum((self.y - self.outer_layer_output)**2) / self.dataset_size
    
    def activate_hidden_layer_nodes(self, input_matrix):
        """ 
        Method that returns the activated outputs from the hidden layer. 

	Args:
	    input_matrix: a numpy array of dimensions ``[input data size X hidden_layer_nodes]``.

        Returns:
            float: a numpy array of activated outputs from this layer ``[input data size X hidden_layer_nodes]``.

        Notes:
            This code will return sigmoid activations from hidden layer.

        """ 
        return  1.0 / (1.0 + (np.exp(-input_matrix)))
        
    def activate_output_layer_nodes(self, input_matrix):
        """ 
        Method that returns the activated outputs from the outer layer. 

	Args:
	    input_matrix: a numpy array of dimensions ``[input data size X outer_layer_nodes]``.

        Returns:
            float: a numpy array of activated outputs from this layer ``[input data size X outer_layer_nodes]``.

        Notes:
            This code will return sigmoid activations from output layer.

        """
        return  1.0 / (1.0 + (np.exp(-input_matrix)))

    def output_layer_derivative(self):
        """ 
        Method that returns the derivative of output layer. 

        Returns:
            float: a numpy array of derivative of outputs from this layer ``[input data size X outer_layer_nodes]``.

        Notes:
            This code will return the derivative of sigmoid function of output layer.

        """ 
        return self.outer_layer_output * (1.0 - self.outer_layer_output)
    
    def hidden_layer_derivative(self):
        """ 
        Method that returns the derivative of hidden layer. 

        Returns:
            float: a numpy array of derivative of outputs from this layer ``[input data size X hidden_layer_nodes]``.

        Notes:
            This code will return the derivative of sigmoid function of hidden layer.

        """ 
        return self.hidden_layer_output * (1.0 - self.hidden_layer_output)

    def feed_forward(self):
        """ 
        Method that propogates outputs from one layer to another till it reaches the output layer.

        Returns:
            Does not return any object.

        Notes:
            This method makes calls to other methods to retrieve activations from each layer with specified activation functions.

        """
        self.hidden_layer_input = np.dot(self.input_layer_output, self.hidden_layer_weights)
        self.hidden_layer_output = self.activate_hidden_layer_nodes(self.hidden_layer_input)
        self.outer_layer_input = np.dot(self.hidden_layer_output, self.outer_layer_weights)
        self.outer_layer_output = self.activate_output_layer_nodes(self.outer_layer_input)
                
    def backpropagate_error(self):
        """ 
        Method that updates the weight matrices by calculating error and gradients at each layer.

        Returns:
            Does not return any object.

        Notes:
            This method makes calls to other methods to retrive gradients of the specified cost function and updates
	    weights according to the learning rate. 

        """
        error =  self.outer_layer_output - self.y
        
        output_error = error * self.output_layer_derivative()
        
        hidden_error_contribution = np.dot(output_error, np.transpose(self.outer_layer_weights))
        hidden_error = hidden_error_contribution * self.hidden_layer_derivative()
    
        self.outer_layer_weights = self.outer_layer_weights - \
            (self.ita * np.dot(np.transpose(self.hidden_layer_output), output_error)) 
        
        self.hidden_layer_weights = self.hidden_layer_weights - \
            (self.ita * np.dot(np.transpose(self.input_layer_output), hidden_error)) 

    def train_model(self):
        """ 
        Method that trains the model with the specified model parameters and convergence threshold.

        Returns:
            Does not return any object.

        Notes:
            This method trains a 3 layer neural network with sigmoid activations at hidden and output layers and converges 
            when the change in error is less than the specified threshold. 

        """
        values = []
        
        prev_error = 9999999999.99
        error = self.cost()

        while((prev_error - error ) > self.threshold ):

            self.feed_forward()
            self.backpropagate_error()
                
            prev_error = error
            error = self.cost()

        self.params.append((self.hidden_layer_weights[:,:-1], self.hidden_layer_weights[:,-1]))
        self.params.append(self.outer_layer_weights)    

    
    def get_params (self):
        """ 
        Method that returns the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return a list of tuples of 
            weights and bias for each layer. Ordering from input to outputt

        """
        return self.params

    def get_predictions (self, x):
        """
        Method returns the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:    
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of 
                            ``x`` 
        Notes:
            Retrieves model parameters and uses those to return predicted labels of the given input as a numpy
array.                            
        """                
        hidden_weight = np.asarray(self.params[0][0])
        hidden_bias = np.asarray(np.reshape(self.params[0][1], (self.params[0][1].shape[0],1)))
        hidden = np.hstack((hidden_weight, hidden_bias))
        
        output_weight = np.asarray(self.params[1])

        input_layer_output = np.concatenate((x, np.ones((len(x),1))), axis = 1)
        
        hidden_layer_input = np.dot(input_layer_output, hidden)
        hidden_layer_output = self.activate_hidden_layer_nodes(hidden_layer_input)
        outer_layer_input = np.dot(hidden_layer_output, output_weight)
        outer_layer_output = self.activate_output_layer_nodes(outer_layer_input)
                
        outer_layer_output[outer_layer_output >=0.5] = 1 
        outer_layer_output[outer_layer_output <0.5] = 0
        
        outer_layer_output = [int(item) for sublist in outer_layer_output for item in sublist]
        return np.asarray(outer_layer_output)

class mlnn(xor_net):
    """
    Inheriting the network above and changing some parameters. 
    """
    def __init__ (self, data, labels):
        super(mlnn,self).__init__(data, labels)
        
        self.hidden_layer_nodes = 15
        
        self.hidden_layer_weights = np.random.randn(self.input_layer_nodes, self.hidden_layer_nodes) - 1
        self.outer_layer_weights = np.random.randn(self.hidden_layer_nodes, self.outer_layer_nodes) - 1 
        
        self.hidden_layer_weights = self.hidden_layer_weights / np.linalg.norm(self.hidden_layer_weights)
        self.outer_layer_weights = self.outer_layer_weights / np.linalg.norm(self.outer_layer_weights)

        self.hidden_layer_input = np.zeros((self.dataset_size, self.hidden_layer_nodes))
        self.hidden_layer_output = np.zeros((self.dataset_size, self.hidden_layer_nodes))

	self.normalize_images()        
        
	self.threshold = 0.00000001 
        self.ita = 0.001  

        self.train_model()

    def normalize_images(self):
        """
        Method returns the normalized matrix representation of input images

        Returns:    
            This function does not return any object
        """
        for img in range(self.input_layer_output.shape[0]):
            self.input_layer_output[img] = self.input_layer_output[img] / self.input_layer_output[img].max()

if __name__ == '__main__':
    pass 

