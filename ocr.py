import cv2
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INPUT_PATH = Path(r'C:\Users\Otto\Desktop\Python\ocr\data\input')
OUTPUT_PATH = Path(r'C:\Users\Otto\Desktop\Python\ocr\data\output')


def get_text_areas(in_path, file_name):
    file_path = str(in_path / file_name)
    img = cv2.imread(file_path)
    #img_final = cv2.imread(file_path)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #lower = np.array([0, 0, 218])
    #upper = np.array([157, 54, 255])
    #mask = cv2.inRange(img2gray, lower, upper)
    
    #convert_bin, grey_scale = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #grey_scale = 255 - grey_scale
    img_size = np.shape(img)
    blocksize = 1 / 8 * img_size[0] / 2 * 2 + 1
    if blocksize <= 1:
        blocksize = img_size[0] / 2 * 2 + 1
    _const = 10
    mask = cv2.adaptiveThreshold(img2gray,
                                 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 21,
                                 5)
    #ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)
    mask = 255 - mask
    #image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    #ret, new_img = cv2.threshold(image_final, 0, 64, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(mask, kernel, iterations=4)  # dilate , more the iteration more the dilation

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        #ar = float(w) / float(h)
        #if ar < 5.:
        if w < 35 or h < 35:
            #cv2.drawContours(dilated, [contour], -1, (0, 0, 0), -1)
            #print(contour)
        # draw rectangle around contour on original image
            #cv2.rectangle(dilated, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.drawContours(dilated, [contour], -1, (0, 0, 0), -1)
        
    # write original image with added contours to disk
    #cv2.imshow('captcha_result', img)
    #cv2.waitKey()
    result = 255 - cv2.bitwise_and(dilated, mask)
    cv2.imwrite(str(OUTPUT_PATH / ('detectable_' + file_name)), result)


def main():
    logger.info('Start exctracting')    

    files = [x for x in os.listdir(INPUT_PATH) if x.endswith('.jpg')]
    for f in files:
        logger.info('Processing file: %s', f)

        file_path = str(INPUT_PATH / f)
        get_text_areas(INPUT_PATH, f)
        
        '''im = cv2.imread(file_path)
        im2 = cv2.imread(file_path, 0)
        
        convert_bin, grey_scale = cv2.threshold(im2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        grey_scale = 255 - grey_scale

        length = np.array(im2).shape[1]//100
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        
        horizontal_detect = cv2.erode(grey_scale, horizontal_kernel, iterations=3)
        hor_line = cv2.dilate(horizontal_detect, horizontal_kernel, iterations=3)

        cv2.imwrite(str(OUTPUT_PATH / ('detectable_' + f)), hor_line)


        ret,thresh_value = cv2.threshold(im2, 180, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((5,5),np.uint8)
        dilated_value = cv2.dilate(thresh_value, kernel, iterations = 1)

        contours, hierarchy = cv2.findContours(dilated_value, 
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        coordinates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            coordinates.append((x, y, w, h))
            #bounding the images
            if y < 50:
                cv2.rectangle(im, (x,y), (x+w,y+h), (0,0,255), 1)


        #plt.imshow(im)
        #cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)
        cv2.imwrite(str(OUTPUT_PATH / ('detectable_' + f)), im)'''


if __name__ == '__main__':
    main()

