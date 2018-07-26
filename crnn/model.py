# coding:utf-8
import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')
import os
import numpy as np
from io import open

from keras.layers import Input,Conv2D,MaxPooling2D,ZeroPadding2D
from keras.models import Model
import keras.backend  as K
from PIL import Image

char = ''
with open('/datasets/text_renderer/data/chars/med.txt',encoding='utf-8') as f:
    for ch in f.readlines():
        ch = ch.strip()
        char += ch
char += '卍'  # '卍' means word not contains in dict
nclass = len(char)
print('nclass:',len(char))
id_to_char = {i:j for i,j in enumerate(char)}

def get_basemodel(modelPath):
    from densent_ocr import densenet
    reload(densenet)

    input = Input(shape=(32,None,1),name='the_input')
    y_pred= densenet.dense_cnn(input,nclass)
    basemodel = Model(inputs=input,outputs=y_pred)
    basemodel.load_weights(modelPath)

    return basemodel

def get_model():
    modelPath =r'../keras_ocr/keras_ocr_model/weights10-densent-01.hdf5'
    if os.path.exists(modelPath):
        basemodel = get_basemodel(modelPath)
    return basemodel

basemodel = get_model()

def predict(im, basemodel=basemodel):
    #im = im.convert('L')
    scale = im.size[1] * 1.0 / 32
    w = im.size[0] / scale
    w = int(w)
    im = im.resize((w, 32), Image.ANTIALIAS)
    img = np.array(im).astype(np.float32) / 255.0-0.5
    X = img.reshape((32, w, 1))
    X = np.array([X])

    y_pred = basemodel.predict(X)
    # y_pred = y_pred[:, 2:, :]
    y_pred = y_pred[:, :, :]
    # out = decode(y_pred)  ##
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :]
    out = u''.join([id_to_char[x] for x in out[0]])
   # out = u''.join([characters[x] for x in out[0]])

    if len(out) > 0:
        while out[0] == u'。':
            if len(out) > 1:
                out = out[1:]
            else:
                break

    return out, im

def decode(pred):
    charactersS = characters + u' '
    t = pred.argmax(axis=2)[0]
    length = len(t)
    char_list = []
    n = len(characters)
    for i in range(length):
        if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
            char_list.append(charactersS[t[i]])
    return u''.join(char_list)
