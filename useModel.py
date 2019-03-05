
# coding: utf-8

# # Mask R-CNN - Inspect Ballon Trained Model
# 
# Code and visualizations to test, debug, and evaluate the Mask R-CNN model.

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../samples/floor")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
import asyncio

import floor

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
print(ROOT_DIR)
# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases


config = floor.FloorConfig()
BALLOON_DIR = os.path.join(ROOT_DIR, "floor_poly") # change this for different directory image to test


# In[3]:


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Notebook Preferences

# In[4]:


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


# In[5]:



# ## Load Validation Dataset

# In[6]:


# Load validation dataset
dataset = floor.FloorDataset()
val_data = dataset.createGen(BALLOON_DIR, "originals") #this function is created to use a generator creating one image for testing at a time


with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


# In[8]:


# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
weights_path = "model/floor20181116T0133/mask_rcnn_floor_0200.h5"

# Or, load the last model you trained
#weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


'''



for i in range(10):
    

    dataset = next(val_data)
    dataset.prepare()
    print("Imagericks: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
    image_id = dataset.image_ids[-1]
    print(image_id)
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,dataset.image_reference(image_id)))

    
    #image = modellib.load_image_gt('data48.jpg', config, single=True)
    
    results = model.detect([image], verbose=1)

    r = results[0]

    masked_image = image.astype(np.uint32).copy()
    N = r['rois'].shape[0]  #find out how many different instances are there
    mask = r['masks'][:, :, 0]

    color = visualize.random_colors(N) #generates random instance colors
    color = color[0]

    image = visualize.apply_mask(masked_image, mask, color)
    cv2.imshow('box', image.astype(np.uint8))
    cv2.waitKey(0)

'''


def processImage(frame):
	image = modellib.load_image_single(frame, config)
	results = model.detect([image], verbose=0)
	r = results[0]
	masked_image = image.astype(np.uint32).copy()
	N = r['rois'].shape[0]  #find out how many different instances are there
	try: #if no mask simply return image frame
		mask = r['masks'][:, :, 0]
		color = visualize.random_colors(N) #generates random instance colors
		color = color[0]
		image = visualize.apply_mask(masked_image, mask, color)
		return image.astype(np.uint8)
	except Exception:
		return image
		


def combine(clip1, clip2):
	import moviepy.editor as mpe
	first = mpe.VideoFileClip(clip1)
	second = mpe.VideoFileClip(clip2)
	final = mpe.concatenate_videoclips([first, second])
	final.write_videofile('testout.mp4', codec = 'libx264')




def read_split_classify(video_in, video_out):
	cap = cv2.VideoCapture(video_in)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

	wait_time = int((1/int(fps)) * 1000)
	#print(height, width)
	# Define the codec and create VideoWriter object
	#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	out = cv2.VideoWriter(video_out,fourcc, fps, (1024, 1024), True)
	framenum = 0
	while(cap.isOpened()):
		cv2.waitKey(wait_time)
		print('PROCESSING FRAME: {}'.format(framenum))
		ret, frame = cap.read()
		if framenum < 0: #650
			framenum+=1
			continue
		if ret:
			try:
				frame = processImage(frame)
			except Exception:
				print("{} Frame has failed video saved".format(framenum))
				break
			out.write(frame)
			#imshow
			cv2.imshow('test', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			framenum += 1

			#if framenum > 10:
				#break

		else:
			break

	# Release everything if job is finished
	cap.release()
	out.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	read_split_classify("test_data_video/skate1.mp4", 'output_skate.mp4')
	#combine('out/output.mp4', 'out/output_final.mp4')