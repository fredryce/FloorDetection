import os
import glob
from PIL import Image
from itertools import tee
from random import shuffle
#import tensorflow as tf
import cv2
import numpy as np
#this tool used to gather the label, original images from each folder and rename it
#https://docs.python.org/3/library/os.path.html#os.path.basename
#eventually use tfrecord format


counter_origin = 0
counter_label = 0

def _bytes_feature(value): #this used to convert numpy arrays to before writing to tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def resize(pathAndFilename, image = True):
	img = Image.open(pathAndFilename)
	if image:
		img = img.resize((200, 200), Image.ANTIALIAS)
		img = img.convert('RGB')
	else:
		img = img.resize((200, 200), Image.ANTIALIAS).convert('1')
	return img

def rename(dir, pattern, titlePattern, image = True, newPath = None):
	global counter_origin, counter_label

	if not 'labels' in os.listdir(newPath):
		os.makedirs(os.path.join(newPath, 'labels'))
	if not 'originals' in os.listdir(newPath):
		os.makedirs(os.path.join(newPath, 'originals'))

  
	for pathAndFilename in glob.iglob(os.path.join(dir, r'*.png')):
		title, ext = os.path.splitext(os.path.basename(pathAndFilename))
		img = Image.open(pathAndFilename)
		img = img.convert('RGB')
		img.save(pathAndFilename.replace('.png', '.jpg'), "JPEG")
		os.remove(pathAndFilename)
		print(pathAndFilename)


	for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
		title, ext = os.path.splitext(os.path.basename(pathAndFilename))
		if image:
			final_image = ''
			if newPath:
				final_image = os.path.join(newPath,'originals')
			else:
				final_labels = dir
			
			new_name = os.path.join(final_image, titlePattern % str(counter_origin) + '.jpg')
			os.rename(pathAndFilename, new_name)
			img = resize(new_name)
			img.save(new_name, 'JPEG')
			counter_origin+=1
		else:
			final_labels = ''
			if newPath:
				final_labels = os.path.join(newPath,'labels')
			else:
				final_labels = dir
			
			new_name = os.path.join(final_labels, titlePattern % str(counter_label) + '.jpg')
			os.rename(pathAndFilename, new_name)
			img = resize(new_name, image = False)
			img.save(new_name, 'JPEG')
			counter_label+=1
		
		
			
def image_pre(directory, newPath = None):

	for direct, direct_name, file in os.walk(directory):
		if 'labels' in direct:
			#print(direct)
			rename(direct, r'*.jpg', r'data%s', image = False, newPath = newPath)
			
		elif 'originals' in direct:
			#print(direct)
			rename(direct, r'*.jpg', r'data%s', newPath = newPath)

#this structure is too sexy!
def test(directory):
	pair = []
	for direct, direct_name, file in os.walk(directory):
		if 'labels' in direct:
			pair.append(direct)
		if 'originals' in direct:
			pair.append(direct)
		if len(pair)== 2:

			result = file_pair_generate(tuple(pair))
			#print('im in directory {} length is {}'.format(pair[0], len(list(result))))
			for items in result:
				yield items


			pair = []
	# in python 3.3 above you can yeild from another function from this function.
	#this fuction is yeiding from the yeild statement in test(directory)

def file_pair_generate(pair):
	result = ((os.path.join(pair[0], file), os.path.join(pair[1], file)) for file in os.listdir(pair[0]))
	return result


def label_manip(image):
	img = Image.open(image)
	img = np.array(img.convert('1')).astype(np.uint8)
	img = 1 - img
	return img



			
def writetotfrecord(data, name = 'floor'): #data format (label, image)
	#filename_pairs = (for direct, directName, file in os.walk(directory) if not 'labels' in direct and not 'originals' in direct)
	#gen = test(directory)
	
	writer = tf.python_io.TFRecordWriter(name)
	for i in range(len(data)):
		if not i % 1000:
			print('Train data: {}/{}'.format(i, len(data)))
		img = cv2.imread(data[i][1])
		#label = cv2.imread(data[i][0])
		label = label_manip(data[i][0])
		feature = {'image': _bytes_feature(img.tostring()), 'label': _bytes_feature(label.tostring())}
		

		example = tf.train.Example(features=tf.train.Features(feature=feature))
		writer.write(example.SerializeToString())
	writer.close()



def main(directory):
	data = list(test(directory))
	shuffle(data)
	print(len(data))
	train = data[0:int(0.8*len(data))]
	#validate = data[int(0.6*len(data)):int(0.8*len(data))]
	testing = data[int(0.8*len(data)):]
	writetotfrecord(train, name = 'floor_train.tfrecords')
	#writetotfrecord(validate, name = 'floor_validate_numpy.tfrecords')
	writetotfrecord(testing, name = 'floor_test.tfrecords')







directory = 'E:/test'
out_dir = 'E:/out/flpoly_usf'




image_pre(directory, out_dir)

#main(out_dir)