from yann.utils.dataset import setup_dataset

def cook_mnist_bg_normalized(  verbose = 1, **kwargs):
    """
    Wrapper to cook mnist dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`

    Notes:
        By default, this will create a dataset that is not mean-subtracted.
    """

    if not 'data_params' in kwargs.keys():

        data_params = {
                   "source"             : 'skdata',
                   "name"               : 'mnist_bg_images',
                   "location"			: '',
                   "mini_batch_size"    : 500,
                   "mini_batches_per_batch" : (80, 20, 20),
                   "batches2train"      : 1,
                   "batches2test"       : 1,
                   "batches2validate"   : 1,
                   "height"             : 28,
                   "width"              : 28,
                   "channels"           : 1  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

    # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : False,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    dataset = setup_dataset(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            verbose = 3)
    return dataset

def cook_mnist_rotated_normalized(  verbose = 1, **kwargs):
        """
    Wrapper to cook mnist rotated dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`

    Notes:
        By default, this will create a dataset that is not mean-subtracted.
    """
    if not 'data_params' in kwargs.keys():

        data_params = {
                    "source"             : 'skdata',
                    "name"               : 'mnist_rotated',
                    "location"			: '',
                    "mini_batch_size"    : 500,
                    "mini_batches_per_batch" : (80, 20, 20),
                    "batches2train"      : 1,
                    "batches2test"       : 1,
                    "batches2validate"   : 1,
                    "height"             : 28,
                    "width"              : 28,
                    "channels"           : 1  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

    # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : False,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    dataset = setup_dataset(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            verbose = 3)
    return dataset

def cook_mnist_noisy_normalized(  verbose = 1, **kwargs):
        """
    Wrapper to cook mnist noisy dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`

    Notes:
        By default, this will create a dataset that is not mean-subtracted.
    """

    if not 'data_params' in kwargs.keys():

        data_params = {
                   "source"             : 'skdata',
                   "name"               : 'mnist_noise3',
                   "location"			: '',
                   "mini_batch_size"    : 500,
                   "mini_batches_per_batch" : (80, 20, 20),
                   "batches2train"      : 1,
                   "batches2test"       : 1,
                   "batches2validate"   : 1,
                   "height"             : 28,
                   "width"              : 28,
                   "channels"           : 1  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

    # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : False,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    dataset = setup_dataset(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            verbose = 3)
    return dataset

if __name__ == '__main__':
    pass
