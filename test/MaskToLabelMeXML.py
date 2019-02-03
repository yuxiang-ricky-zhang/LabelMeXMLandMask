'''
Adapted from https://github.com/sacmehta/SegmentationMaskToXML
'''

import cv2
import numpy as np
import os
import glob
from argparse import ArgumentParser
import copy
import xml.etree.cElementTree as ET
import datetime

idToClassMap = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light',
                7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
                14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle', 19: 'undefined'}
idToColor = {0: '#804080', 1: '#F423E8', 2: '#464646', 3: '#66669C', 4: '#BE9999', 5: '#999999', 6: '#FAAA1E',
                7: '#DCDC00', 8: '#6B8E23', 9: '#97FC97', 10: '#4682B4', 11: '#DC143C', 12: '#FF0000', 13: '#00008E',
                14: '#000046', 15: '#003C64', 16: '#005064', 17: '#0000E6', 18: '#770B20', 19: '#000000'}

username = 'sachin'

def main(args):
    rgb_list = []
    if not os.path.isdir(args.rgb_dir):
        print('RGB Image directory does not exist.')
        exit(-1)

    rgb_search = args.rgb_dir + os.sep + '*.' + args.rgb_format
    rgb_list = glob.glob(rgb_search)
    if len(rgb_list) == 0:
        print('RGB image directory (' + rgb_search + ' ) does not contain any images with format ' + args.rgb_format)
        print('Exiting!!')
        exit(-1)

    #create output directory
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    for i, rgb_file in enumerate(rgb_list):
        xml_root = ET.Element("annotation")
        mask_file_name = rgb_file.replace(args.rgb_dir, args.mask_dir).replace(args.rgb_format, 'png') #mask image should always be png format
        file_name = rgb_file.split('/')[-1]
        folder_name = args.rgb_dir

        fName = ET.SubElement(xml_root, "filename")
        fName.text = file_name

        folName = ET.SubElement(xml_root, "folder")
        folName.text = folder_name

        source = ET.SubElement(xml_root, "source")
        sourceIM = ET.SubElement(source, "sourceImage")
        sourceIM.text = "The MIT-CSAIL database of objects and scenes"

        sourceAnn = ET.SubElement(source, "sourceAnnotation")
        sourceAnn.text = "LabelMe Webtool"

        mask = cv2.imread(mask_file_name, 0)
        mask_val = np.unique(mask)



        # remove the undifined
        if 19 in mask_val:
            index = np.argwhere(mask_val==19)
            mask_val = np.delete(mask_val, index)

        cols, rows = mask.shape[:2]

        imSize = ET.SubElement(source, "imagesize")
        nrows = ET.SubElement(imSize, "nrows")
        nrows.text = str(rows)
        ncols = ET.SubElement(imSize, "ncols")
        ncols.text = str(cols)


        for gId, val in enumerate(mask_val):
            # find the countours
            class_mask = np.zeros_like(mask, dtype=np.uint8)
            class_mask[mask == val] = 1
            contours, hierarchy = cv2.findContours(class_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Remove small polygons
            if (val == 0 or val == 2 or val == 8 or val == 9 or val == 10):
            	perimeter_thres = 80
            else:
            	perimeter_thres = 20

            for i, cnt in enumerate(contours):
            	if (cv2.arcLength(cnt,True) > perimeter_thres):
	                object = ET.SubElement(xml_root, "object")
	                objName = ET.SubElement(object, "name")
	                objName.text = idToClassMap[val]
	                objDel = ET.SubElement(object, "deleted")
	                objDel.text = 0
	                objVerif = ET.SubElement(object, "verified")
	                objVerif.text = 0
	                objOcc = ET.SubElement(object, "occluded")
	                objOcc.text = 'no'
	                objAtt = ET.SubElement(object, "attributes")
	                objHParts = ET.SubElement(object, "hasparts")
	                objParts = ET.SubElement(object, "parts")
	                objIsPartOf = ET.SubElement(objParts, "ispartof")
	                objIsPartOf = ET.SubElement(objParts, "ispartof")

	                date = ET.SubElement(object, "date")
	                now = datetime.datetime.now()
	                date.text = str(now)

	                id = ET.SubElement(object, "id")
	                id.text = str(i)

	                polygon = ET.SubElement(object, "polygon")
	                uname = ET.SubElement(polygon, "username")
	                uname.text = username

	                #contour approximation
	                if (len(cnt) > 50):
	                	epison = 0.001*cv2.arcLength(cnt,True)
	                	cnt = cv2.approxPolyDP(cnt,epison,True)

	                for j in range(len(cnt)):
	                    x = cnt[j][0][0]
	                    y = cnt[j][0][1]
	                    point = ET.SubElement(polygon, "pt")
	                    ptX = ET.SubElement(point, "x")
	                    ptX.text = str(x)

	                    ptY = ET.SubElement(point, "y")
	                    ptY.text = str(y)


        tree = ET.ElementTree(xml_root)
        tree.write(args.out_dir + os.sep + file_name.split('.')[0] + '.xml', encoding="utf-8",
                   xml_declaration=True, method='xml')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--rgb_dir', default="./rgb", help='RGB Images')
    parser.add_argument('--rgb_format', default="jpg", help='RGB Image format')
    parser.add_argument('--mask_dir', default="./mask", help='Segmentation Mask Images')
    parser.add_argument('--out_dir', default="./xml/", help='Output directory containing XML Files.')

    main(parser.parse_args())
