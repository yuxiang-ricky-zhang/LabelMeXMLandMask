import xml.etree.ElementTree as ET 
import numpy as np
import cv2

idToClassMap = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light',
                7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
                14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle', 19: 'undefined'}

classToidMap = {v: int(k) for k, v in idToClassMap.items()}


tree = ET.parse('img1_test.xml')

root = tree.getroot() 


size = root.findall('imagesize')

nrows = int(size[0].findall('nrows')[0].text)
ncols = int(size[0].findall('ncols')[0].text)

mask = np.full([nrows,ncols], 19, dtype = np.uint8)

object_cnt = 0
for instance in root.iter('object'):
	object_cnt += 1

	if (instance.findall('deleted')[0].text) == '0':

		classname = instance.findall('name')[0].text

		cnt_points = []

		for pt in instance.iter('pt'):

			for x in pt.findall('x'):
				# print("x:", x.text)
				ptx = int(x.text)

			for y in pt.findall('y'):
				# print("y:", y.text)
				pty = int(y.text)

			cnt_points.append([ptx, pty])


	pts = np.asarray([cnt_points], dtype=np.int32)


	cv2.fillPoly(img= mask, pts = pts,color = classToidMap[classname])

cv2.imwrite('./mask.png',mask)



