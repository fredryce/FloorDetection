import glob
import os
from PIL import Image
import numpy as np
import cv2
def findMissing():
	array = [0]*8061
	#print(len(array))
	for pathAndFilename in glob.iglob(os.path.join('./labels', r'*.jpg')):
		base = int(os.path.basename(pathAndFilename).split('.')[0])
		array[base] = 1
		#print(base)
	for i in range(len(array)):
		if array[i] != 1:
			print(i)




def add_black(num):
	for i in num:
		image = np.zeros((720, 1280), dtype=np.uint8)
		name = os.path.join('./labels', str(i) + '.jpg')
		cv2.imwrite(name, image)

def flip(smooth = False, flip = True):
	for pathAndFilename in glob.iglob(os.path.join('./labels', r'*.jpg')):
	    img = Image.open(pathAndFilename)
	    img = np.array(img.convert('1')).astype(np.uint8)

	    #img = cv2.bitwise_not(img)
	    if flip:
	    	img = (1 - img)*255  #this is for flipping for the usf dataset
	    else:
	    	img = img * 255
	    #img.save(pathAndFilename, "JPEG")
	    if smooth:
	    	img = cv2.medianBlur(img,35)
	    cv2.imwrite(pathAndFilename, img)
	    print(img)
flip(smooth=True, flip = False)


#add_black([x for x in range(3356,3467)])
#findMissing()