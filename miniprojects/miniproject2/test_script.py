# will ``sample_submission`` to your submission filename.

# from asurite_lastname.networks import xor_net, mlnn
from amadan4_madan.networks import xor_net, mlnn
from dataset import xor, waldo
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

def test_xor():
    """
    This method will run the test on xor dataset.
    """
    dg = xor() # Initialize a dataset creator
    training_data, training_labels = dg.query_data(samples = 100) 

    n = xor_net(training_data, training_labels)  # This call should return a net object
    params = n.get_params()    # This call should reaturn parameters of the model that are 
                               # fully trained.

    testing_data, testing_labels = dg.query_data(samples = 100)  # Create a random testing dataset.
    predictions = n.get_predictions(testing_data) # This call should return predictions.

    acc = accuracy(testing_labels, predictions)
    print "Accuracy of predictions on XOR data = " + str(acc) + "%"
    return acc

def test_waldo():
    """
    This method will run the test on waldo dataset. 
    """
    dg = waldo() # Initialize a dataset creator
    training_data, training_labels = dg.query_data(samples = 100) 

    n = mlnn(training_data, training_labels)  # This call should return a net object that is trained.
    params = n.get_params()    # This call should reaturn parameters of the model that are 
                               # fully trained.

    testing_data, testing_labels = dg.query_data(samples = 100) 
    predictions = n.get_predictions(testing_data) # This call should return predictions.

    acc = accuracy(testing_labels, predictions)
    print "Accuracy of predictions on waldo data = " + str(acc) + "%" 
    return acc

if __name__ == '__main__':
    
    # Part 1 of the project. 
    xor_acc = test_xor()
    
    # Part 2 of the project.
    waldo_acc = test_waldo()   

    weight = np.random.uniform(low = 0.3, high = 0.7)
    # This means that if you score 0.9 on random weighted average,
    #  I will give you full grade essentially...

    mix = min(100, weight* xor_acc + (1-weight) * waldo_acc + 10)

    # 3+ points minimum for either getting a code that works, or submitting the same code as it is.
    # Grader will check if code produces random number like what the sample does, then he will give 
    # you a zero. 
    grade = 2 + (mix / 50)    
    print "Grade = " + str( grade )

