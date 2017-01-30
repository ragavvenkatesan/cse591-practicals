# will ``sample_submission`` to your submission filename.

from sample_submission import regressor
from nnet import dataset_generator, rmse

if __name__ == '__main__':
    
    dg = dataset_generator() # Initialize a dataset creator
    data_train = dg.query_data(samples = 5000) # Create a random training dataset.

    r = regressor(data_train)  # This call should return a regressor object that is fully trained.
    params = r.get_params()    # This call should reaturn parameters of the model that are 
                               # fully trained.

    data_test = dg.query_data(samples = 5000)  # Create a random testing dataset.
    predictions = r.get_predictions(data_test[0]) # This call should return predictions.

    print "Rmse error of predictions = " + str(rmse(data_test[1], predictions))