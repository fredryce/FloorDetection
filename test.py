import tensorflow as tf
import numpy as np
from PIL import Image
import os,sys
from os import path as osp
from glob import glob
sys.path.append(os.path.abspath('./../../mrcnn'))
import model as modellib
from config import Config
import skimage.io
import utils
import visualize
import matplotlib.pyplot as plt
import math
import random
import cv2
slim = tf.contrib.slim

inference_config = Config()

def main(argv=None):
    ROOT_DIR = os.getcwd()

    with tf.gfile.FastGFile("./pb_out_new.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    print('Graph loaded.')

    with tf.Session() as sess:
        #IMAGE_DIR = os.path.join(ROOT_DIR, 'images') #image of the size defined in the config
        #file_names = next(os.walk(IMAGE_DIR))[2]
        #filename = random.choice(file_names)
        #print('choose image file is', filename)
        image = skimage.io.imread("./151.jpg")
#       os.system('eog %s &'%(os.path.join(IMAGE_DIR,filename)))
        print(image.shape)
        images = [image]
        print("Processing {} images".format(len(images)))
        for im in images:
             modellib.log("image", im)
        print('RGB image loaded and preprocessed.')


        molded_images, image_metas, windows = mold_inputs(images)
        print(molded_images.shape)
        
        image_shape = molded_images[0].shape
        # Anchors
        anchors = get_anchors(image_shape, inference_config)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        inference_config.BATCH_SIZE = 1
        image_anchors = np.broadcast_to(anchors, (inference_config.BATCH_SIZE,) + anchors.shape)
        print('anchors shape is', image_anchors.shape, image_anchors.dtype)

        img_ph = sess.graph.get_tensor_by_name('input_image:0')
        print(img_ph)
        img_anchors_ph = sess.graph.get_tensor_by_name('input_anchors:0')
        print(img_anchors_ph)
        img_meta_ph = sess.graph.get_tensor_by_name('input_image_meta:0')
        print(img_meta_ph)
        detectionsT = sess.graph.get_tensor_by_name('output_detections:0')
        print('Found ',detectionsT)
        mrcnn_classT = sess.graph.get_tensor_by_name('output_mrcnn_class:0')
        print('Found ',mrcnn_classT)
        mrcnn_bboxT = sess.graph.get_tensor_by_name('output_mrcnn_bbox:0')
        print('Found ', mrcnn_bboxT)
        mrcnn_maskT = sess.graph.get_tensor_by_name('output_mrcnn_mask:0')
        print('Found ', mrcnn_maskT)
        roisT = sess.graph.get_tensor_by_name('output_rois:0')
        print('Found ', roisT)
        
        np.set_printoptions(suppress=False,precision=4)
        print('Windows', windows.shape,' ',windows)
        detections = sess.run(detectionsT, feed_dict={img_ph: molded_images, img_meta_ph: image_metas, img_anchors_ph:image_anchors})
        #print('Detections: ',detections[0].shape, detections[0])
        mrcnn_class = sess.run(mrcnn_classT, feed_dict={img_ph: molded_images, img_meta_ph: image_metas, img_anchors_ph:image_anchors})
        #print('Classes: ',mrcnn_class[0].shape, mrcnn_class[0])
        mrcnn_bbox = sess.run(mrcnn_bboxT, feed_dict={img_ph: molded_images, img_meta_ph: image_metas, img_anchors_ph:image_anchors})
        #print('BBoxes: ',mrcnn_bbox[0].shape, mrcnn_bbox[0])
        mrcnn_mask = sess.run(mrcnn_maskT, feed_dict={img_ph: molded_images, img_meta_ph: image_metas, img_anchors_ph:image_anchors})
        #print('Masks: ',mrcnn_mask[0].shape )#, outputs1[0])
        rois = sess.run(roisT, feed_dict={img_ph: molded_images, img_meta_ph: image_metas, img_anchors_ph:image_anchors})
        #print('Rois: ',rois[0].shape, rois[0])

        
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                unmold_detections(detections[i], mrcnn_mask[i],
                                  image.shape, molded_images[i].shape,
                                  windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })

        r = results[0]
        print(results)
        #print('result is', r)
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])#, ax=get_ax())
    print('Done')

    print("mask shape", r['masks'].shape)
    #image = cv2.imread(testImage)
    N = r['rois'].shape[0]  #find out how many different instances are there
    try: #if no mask simply return image frame
        mask = r['masks'][:,:,0]
        color = visualize.random_colors(N) #generates random instance colors
        color = color[0]
        image = visualize.apply_mask(image, mask, color)
    except Exception as e:
        print(e)
    cv2.imshow('test', image)
    cv2.waitKey(0)
    return 0

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.
    
    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])

def get_anchors(image_shape, config):
    """Returns anchor pyramid for the given image size."""
    backbone_shapes = compute_backbone_shapes(config, image_shape)
    # Cache anchors and reuse if image shape is the same
    _anchor_cache = {}
    if not tuple(image_shape) in _anchor_cache:
        # Generate Anchors
        a = utils.generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE)
        # Keep a copy of the latest anchors in pixel coordinates because
        # it's used in inspect_model notebooks.
        # TODO: Remove this after the notebook are refactored to not use it
        anchors = a
        # Normalize coordinates
        _anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
    return _anchor_cache[tuple(image_shape)]

def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w,h = im.size
    for i in range(w):
        for j in range(h):
            r,g,b = im.getpixel((i,j))
            if r != g != b: return False
    return True


def mold_inputs(images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, corp = utils.resize_image(
                image,
                min_dim=inference_config.IMAGE_MIN_DIM,
                min_scale=inference_config.IMAGE_MIN_SCALE,
                max_dim=inference_config.IMAGE_MAX_DIM,
                mode=inference_config.IMAGE_RESIZE_MODE)

            print(image.shape)
            print('Image resized at: ', molded_image.shape)
            print(window)
            print(scale)
            """Takes RGB images with 0-255 values and subtraces
                   the mean pixel and converts it to float. Expects image
                   colors in RGB order."""
            molded_image = mold_image(molded_image, inference_config)
            print('Image molded')
            #print(a)
            """Takes attributes of an image and puts them in one 1D array."""
            inference_config.NUM_CLASSES = 81
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([2], dtype=np.int32))
            print('Meta of image prepared')
            image_anchor = [] # TODO
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

def mold_image(images, config):
    return images.astype(np.float32) - config.MEAN_PIXEL

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta

def unmold_detections(detections, mrcnn_mask, original_image_shape, image_shape, window):
    """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]

    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    window = utils.norm_boxes(window, image_shape[:2])
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        # Convert neural network mask to full size mask
        full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
        if full_masks else np.empty(masks.shape[1:3] + (0,))

    return boxes, class_ids, scores, full_masks

if __name__ == '__main__':
    tf.app.run()