# imports
import argparse
from .train_cnn import train_cnn
from .train_segmentation import train_segmentation
from .train_sklearn import train_sklearn


# global model mapping calling specific training function
model_mapping = {
    'efficient_net_b7': 'cnn',
    'efficient_net_v2s': 'cnn',
    'enet': 'seg',
    'inception_v3': 'cnn',
    'knn': 'sklearn',
    'resnet50': 'cnn',
    'rf': 'sklearn',
    'seq': 'cnn',
    'simple_seq': 'cnn',
    'svm': 'sklearn',
    'unet': 'seg',
    'vgg16': 'cnn'
}


# main
if __name__ == '__main__':
    # defining argument parser
    parser = argparse.ArgumentParser(description='process the inputs')
    parser.add_argument(
        '--model',
        type=str,
        help='which model would you like to run',
        default='inception_v3'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        help='how many epochs (if applicable)',
        default=3
    )
    parser.add_argument(
        '--verbose', 
        type=int, 
        help='0,1,2',
        default=1
    )

    # extracting values from argument parser
    args = parser.parse_args()
    _model = args.model
    _epochs = args.epochs
    _verbose = args.verbose

    if _model not in model_mapping.keys():
        raise Exception('not a valid model')

    # training the individual models
    im_size = (448,448,3)
    BATCH_SIZE = 16
    model, traingen, valgen = None, None, None

    if model_mapping[_model] == 'cnn':
        model, traingen, valgen = train_cnn(
            _model, 
            _verbose,
            _epochs,
            BATCH_SIZE=BATCH_SIZE, 
            im_size=im_size,
        )
    elif model_mapping[_model] == 'seg':
        model, traingen, valgen = train_segmentation(
            _model, 
            _verbose, 
            _epochs, 
            BATCH_SIZE=BATCH_SIZE, 
            im_size=im_size, 
        )
    elif model_mapping[_model] == 'sklearn':
        model, traingen, valgen = train_sklearn(
            _model, 
            _verbose,
        )

