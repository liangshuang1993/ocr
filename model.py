# coding:utf-8
##添加文本方向 检测模型，自动检测文字方向，0、90、180、270
from math import *

import cv2
import os
import shutil
import numpy as np
from PIL import Image
import sys

# sys.path.append("crnn")
# from angle.predict import predict as angle_detect  ##文字方向检测


from ctpn.text_detect import text_detect
from crnn.model import predict as crnn_predict

import draw

RESULTS = 'results'
RESULTS_CTPN = os.path.join(RESULTS, 'ctpn')
if os.path.exists(RESULTS):
    shutil.rmtree(RESULTS)
os.makedirs(RESULTS)
os.makedirs(RESULTS_CTPN)

def ruihua(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # retval2,thre2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  
    # return thre2
    imgAdapt=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
    return imgAdapt

def crnnRec(imname, im, text_recs, ocrMode='keras', adjust=False):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im:Array
    @@text_recs:text box

    """
    index = 0
    results = {}
    xDim, yDim = im.shape[1], im.shape[0]

    os.makedirs(os.path.join(RESULTS_CTPN, imname.split('.')[0]))

    for index, rec in enumerate(text_recs):
        results[index] = [
            rec,
        ]
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2),
                   min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  ##图像倾斜角度

        partImg = dumpRotateImage(im, degree, pt1, pt2, pt3, pt4)

        min_x = min(rec[0], rec[2], rec[4], rec[6])
        min_y = min(rec[1], rec[3], rec[5], rec[7])
        max_x = max(rec[0], rec[2], rec[4], rec[6])
        max_y = max(rec[1], rec[3], rec[5], rec[7])
        partImg = im[min_y: max_y, min_x: max_x, :]
        # 根据ctpn进行识别出的文字区域，进行不同文字区域的crnn识别
        image = Image.fromarray(partImg).convert('L')
        # image = Image.fromarray(ruihua(partImg))
        image.save(os.path.join(RESULTS_CTPN, imname.split('.')[0], str(index) + '.png'))
        # cv2.imwrite(os.path.join(RESULTS_CTPN, imname.split('.')[0], str(index) + '.png'), image)
        # 进行识别出的文字识别
        sim_pred = crnn_predict(image)
        results[index].append(sim_pred)  ##识别文字

    return results


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(
        img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation,
                                  np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation,
                                  np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])):min(ydim - 1, int(pt3[1])),
                         max(1, int(pt1[0])):min(xdim - 1, int(pt3[0]))]
    # height,width=imgOut.shape[:2]
    return imgOut


def model(im_name, img, model='keras', adjust=True, detectAngle=False):
    """
    @@param:img,
    @@param:model,选择的ocr模型，支持keras\pytorch版本
    @@param:adjust 调整文字识别结果
    @@param:detectAngle,是否检测文字朝向
    
    """
    angle = 0
    original_img = img
    # 进行图像中的文字区域的识别
    text_recs, tmp, img=text_detect(os.path.join(RESULTS_CTPN, im_name), img)
    # 识别区域排列
    # text_recs = sort_box(text_recs)
    # 
     
    results = crnnRec(im_name, original_img, text_recs, model, adjust=adjust)

    # draw new image
    new_img = np.ones(original_img.shape) * 255
    color_ = (0,0,0)
    text_size = 20
    ft = draw.put_chinese_text('/datasets/text_renderer/data/fonts/chn/songti.ttf')

    for result in results.items():
        box, b = result[1]
        pos = (box[0], box[1])
        new_img = ft.draw_text(new_img, pos, b[0], text_size, color_)
    cv2.imwrite(os.path.join(RESULTS, im_name), new_img)      
    
    return tmp, angle


def sort_box(box):
    """
    对box排序,及页面进行排版
    text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
    """

    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box


