# will ``sample_submission`` to your submission filename.

from sample_submission import xor_net, mlnn
from dataset import xor
import numpy as np

def accuracy ( labels,  predictions ): 
    """
    This function returns a count of the number of uneuqal elements in the two input arrays.
    
    Args:
        labels: first input ndarray
        predictions: second input ndarray

    Returns: 
        numpy float: accuracy in predictions

    Notes:
        The grade that you will get will depend on this output. The lower this value, the higher 
        your grade.
    """         
    return (np.sum(np.asarray(predictions == labels, dtype ='int'),axis = 0) / float(labels.shape[0])) * 100

if __name__ == '__main__':
    
    # Part 1 of the project. 
    dg = xor() # Initialize a dataset creator
    training_data, training_labels = dg.query_data(samples = 5000) 
    dg.plot(training_data, training_labels)

    n = xor_net(training_data, training_labels)  # This call should return a net object that is trained.
    params = n.get_params()    # This call should reaturn parameters of the model that are 
                               # fully trained.

    testing_data, testing_labels = dg.query_data(samples = 5000)  # Create a random testing dataset.
    predictions = n.get_predictions(testing_data) # This call should return predictions.

    print "Accuracy of predictions on XOR data = " + str(accuracy(testing_labels, predictions)) + "%"

    
    # Part 2 of the project.
    dg = mystery() # Initialize a dataset creator
    training_data, training_labels = dg.query_training_data() 

    n = mlnn(training_data, training_labels)  # This call should return a net object that is trained.
    params = n.get_params()    # This call should reaturn parameters of the model that are 
                               # fully trained.

    testing_data, testing_labels = dg.query_testing_data() 
    predictions = n.get_predictions(testing_data) # This call should return predictions.

    print "Accuracy of predictions on mystery data = " + str(accuracy(testing_labels, predictions)) + "%" 
     