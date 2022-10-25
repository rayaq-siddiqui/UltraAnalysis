# uvicorn api:app --reload
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import json
# cnn models
from models import (
    efficient_net_b7,
    efficient_net_v2s,
    inception_v3,
    resnet50,
    seq_model,
    simple_seq_model,
    vgg16
)
# seg models
from models import (
    unet,
    enet,
)
# sklearn models
from models import (
    svm,
    rf,
    knn
)


# defining the app
app = FastAPI()


# define the routing
@app.get('/')
async def prediction_endpoint():
    return {'hello': 'world'}


@app.get('/seg/')
async def segmentation_model(data: UploadFile = File(...)):
    # using best model for now
    model = enet.ENet(
        n_classes=1, 
        input_height=im_size[0], 
        input_width=im_size[1]
    )
    model.load_weights('checkpoints/enet_best_0.01815/')
    return {'hello': 'world'}


@app.get('/cnn/')
async def cnn_model(data: UploadFile = File(...)):
    return {'hello': 'world'}


@app.get('/sklearn/')
async def sklearn_model(data: UploadFile = File(...)):
    return {'hello': 'world'}
