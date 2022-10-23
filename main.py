# imports
import argparse
from train_cnn import train_cnn
from train_segmentation import train_segmentation
from train_sklearn import train_sklearn
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score


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


def fn(x):
    return 0. if x < 0.5 else 1.
fn = np.vectorize(fn)


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
    parser.add_argument(
        '--weights', 
        type=bool, 
        help='load weights (bool)',
        default=False
    )

    # extracting values from argument parser
    args = parser.parse_args()
    _model = args.model
    _epochs = args.epochs
    _verbose = args.verbose
    _weights = args.weights

    if _model not in model_mapping.keys():
        raise Exception('not a valid model')

    # training the individual models
    im_size = (448,448,3)
    BATCH_SIZE = 16
    model, traingen, valgen = None, None, None

    if model_mapping[_model] == 'cnn':
        print('cnn model')
        model, traingen, valgen = train_cnn(
            _model, 
            _verbose,
            _epochs,
            BATCH_SIZE=BATCH_SIZE, 
            im_size=im_size,
        )

        val_X, val_y = valgen.get_all_data()
        pred = model.predict(np.array(val_X))
        acc = val_y

        pred_new = []
        for p in pred:
            # m = max(p)
            pred_new.append(np.argmax(p))
        pred_new = np.array(pred_new)
        pred = pred_new

        f1 = f1_score(acc,pred, average='micro')
        print(f"f1_score: {f1}")

    elif model_mapping[_model] == 'seg':
        print('segmentation model')
        model, traingen, valgen = train_segmentation(
            _model, 
            _verbose, 
            _epochs, 
            BATCH_SIZE=BATCH_SIZE, 
            im_size=im_size, 
            load_weights=_weights
        )

        # img,mask,predict
        test = valgen.__getitem__(0)
        img = test[0][8]
        mask = test[1][8]
        pred = model.predict(tf.convert_to_tensor([img]))[0]        
        pred = fn(pred)

        print(np.unique(pred))

        # display images
        plt.imshow(img, interpolation='nearest')
        plt.show()
        plt.imshow(mask, cmap='gray')
        plt.show()
        plt.imshow(pred, cmap='gray')
        plt.show()

    elif model_mapping[_model] == 'sklearn':
        print('sklearn model')
        model, traingen, valgen = train_sklearn(
            _model, 
            _verbose,
        )

        val_X, val_y = valgen.get_all_data()
        pred = model.predict(val_X)
        acc = val_y

        f1 = f1_score(acc, pred, average='micro')
        print(f"f1_score: {f1}")

    print('model demo complete')
