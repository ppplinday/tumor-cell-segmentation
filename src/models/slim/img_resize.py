import glob
import os
import sys
import cv2


raw_dir = '/disk1/medical/images/'

ls = os.listdir(raw_dir)
for l in ls:
	path = os.path.join(raw_dir, l)
	files = os.listdir(path)
	for image in files:
		img_path = path + image
		raw_img = cv2.imread(img_path, -1)
		img = cv2.resize(raw_img, (224, 224), interpolation=cv2.INTER_AREA)
		img_path = img_path[:-4]
		img_path = img_path + '.jpg'
		cv2.imwrite(image, img)

