# -*- coding: utf-8 -*-  

import numpy as np
from PIL import Image
from glob import glob
import model
import time
import os
# ces


if __name__ == '__main__':
    images = glob('/opt/text/pics/*.*')
    for image in images:
        if '6.jpg' not in image:
            continue
    #images = ['qingxi.jpg']
        im_name = os.path.basename(image)
        im = Image.open(image)
        img = np.array(im.convert('RGB'))
        t = time.time()
        img,angle = model.model(im_name, img,model='keras',detectAngle=False) ##if model == keras ,you should install keras
        print "It takes time:{}s".format(time.time()-t)
        print "---------------------------------------"
        # print "图像的文字朝向为:{}度\n".format(angle),"识别结果:\n"
        # for key in result:
        #     print result[key][1]
        # Image.fromarray(img)
