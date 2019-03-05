'''
this file convert from keras h5 model to tensorflow pb
'''

import tensorflow as tf
import keras
import os, sys
ROOT_DIR = os.path.abspath("./../../")
sys.path.append(ROOT_DIR)
import floor
import mrcnn.model as modellib
from mrcnn.config import Config 
from keras import backend as K
from tensorflow.python.framework import graph_util
save_model = False

if save_model:

	weights_path = os.path.join(ROOT_DIR, "mask_rcnn_floor_0042.h5")
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	config = floor.FloorConfig()
	with tf.device('/cpu:0'):
	    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)

	#model = keras.models.load_model("mask_rcnn_floor_0042.h5")
	model.load_weights(weights_path, by_name=True)

	save_to_ckpt = False

	if save_to_ckpt:
		saver = tf.train.Saver()
		sess = keras.backend.get_session()
		save_path = saver.save(sess, os.path.join(ROOT_DIR, "model.ckpt"))
	else:
		#model.ModelCheckpoint(os.path.join(ROOT_DIR, "keras_all_model.h5"), verbose=0, save_weights_only=False)
		model.keras_model.save(os.path.join(ROOT_DIR, "keras_all_model.h5"))
else:
	class InferenceConfig(Config):
	    # Set batch size to 1 since we'll be running inference on
	    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	    NAME= "myConfig"
	    GPU_COUNT = 1
	    NUM_CLASSES = 2
	    IMAGES_PER_GPU = 1
	    IMAGE_MIN_DIM = 512 
	    IMAGE_MAX_DIM = 512 
	    USE_MINI_MASK = False
	    RESNET_ARCHITECTURE = "resnet50"
	    DETECTION_MAX_INSTANCES = 50
	    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
	    TRAIN_ROIS_PER_IMAGE = 100
	    STEPS_PER_EPOCH = 200
	    VALIDATION_STEPS = 5

	myconfig = InferenceConfig()
	myconfig.display()
	model = modellib.MaskRCNN(mode="inference",config=myconfig,model_dir="./log")
	# Get path to saved weights. Either set a specific path or find last trained weights
	model_filepath = "./model/floor20181116T0133/mask_rcnn_floor_0043.h5"
	#model_filepath = model_filepath if model_filepath else model.find_last()[1]
	#print(model_filepath)
	# Load trained weights (fill in path to trained weights here)
	assert model_filepath, "Provide path to trained weights"
	model.load_weights(model_filepath, by_name=True)
	print("Model loaded.")
	model_keras= model.keras_model
	# All new operations will be in test mode from now on.
	K.set_learning_phase(0)
	# Create output layer with customized names
	num_output = 7
	pred_node_names = ["detections", "mrcnn_class", "mrcnn_bbox", "mrcnn_mask",
                       "rois", "rpn_class", "rpn_bbox"]
	pred_node_names = ["output_" + name for name in pred_node_names]
	print(pred_node_names)
	pred = [tf.identity(model_keras.outputs[i], name = pred_node_names[i])
        for i in range(num_output)]
	sess = K.get_session()
	# Get the object detection graph
	od_graph_def = graph_util.convert_variables_to_constants(sess,
                                                         sess.graph.as_graph_def(),
                                                         pred_node_names)
	model_dirpath = os.path.dirname(model_filepath)
	#pb_filepath = os.path.join(model_dirpath, filename)
	#print('Saving frozen graph {} ...'.format(os.path.basename(pb_filepath)))

	frozen_graph_path = "./pb_out_new.pb"
	with tf.gfile.GFile(frozen_graph_path, 'wb') as f:
		f.write(od_graph_def.SerializeToString())
	print('{} ops in the frozen graph.'.format(len(od_graph_def.node)))
	#print('pb file saved at', model_dirpath)

