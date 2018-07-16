# coding:utf-8
import time
from glob import glob
import os

import numpy as np
from PIL import Image

import model
import find_bbox
# ces


if __name__ == '__main__':
    images = glob('/opt/text/pics/*.*')
    # paths = glob('./test/*.*')
    for image in images:
        if '11' not in image:
            continue
        # im = Image.open(image)
        image = 'ccc.PNG'
        # find bounding box of font aera
        im = find_bbox.process_image(image, '', save_flag=False)
        im.save('ttt.png')
        w, h = im.size
        for i in range(3):

            if i == 0:
                box = (0, 0, w / 2, h)
            elif i == 1:
                box = (w / 4, 0, 3 * w / 4, h)
            elif i == 2:
                box = (w / 2, 0, w, h)
            tmp_img = im.crop(box)
            tmp_img.save('testaa{}.png'.format(str(i)))
            # tmp_img = tmp_img.resize((2000, 2000 * h / w), Image.ANTIALIAS)
            # tmp_img.save('test{}.png'.format(str(i)))

            print tmp_img.size
            
            img = np.array(im.convert('RGB'))
            img,angle = model.model(i, os.path.basename(image), img ,model='keras',detectAngle=False) ##if model == keras ,you should install keras
            print "---------------------------------------"
