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
async def prediction_endpoint():
    return {'hello': 'world'}


@app.get('/cnn/')
async def prediction_endpoint():
    return {'hello': 'world'}


@app.get('/sklearn/')
async def prediction_endpoint():
    return {'hello': 'world'}
