import tensorflow as tf
import os
import imageio
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize
import mrcnn.utils as utils
import numpy as np
import math
import cv2
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

inference_config = InferenceConfig()
    




def mold_inputs(images):
    molded_images = []
    image_metas = []
    windows = []
    #print('IMAGE_PADDING: ',inference_config.IMAGE_PADDING)
    for image in images:
        # Resize image to fit the model expected size
        # TODO: move resizing to mold_image()
        molded_image, window, scale, padding, _ = utils.resize_image(
            image,
            min_dim=inference_config.IMAGE_MIN_DIM,
            min_scale=inference_config.IMAGE_MIN_SCALE,
            max_dim=inference_config.IMAGE_MAX_DIM,
            mode=inference_config.IMAGE_RESIZE_MODE
            )
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
        image_meta = compose_image_meta(
            0, image.shape, molded_image.shape, window, scale,
        np.zeros([inference_config.NUM_CLASSES], dtype=np.int32))
        print('Meta of image prepared')
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

def compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(original_image_shape)+
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale]+
        list(active_class_ids)  # size=num_classes
    )
    return meta
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

def unmold_detections(detections, mrcnn_mask, image_shape, window):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    detections: [N, (y1, x1, y2, x2, class_id, score)]
    mrcnn_mask: [N, height, width, num_classes]
    image_shape: [height, width, depth] Original size of the image before resizing
    window: [y1, x1, y2, x2] Box in the image where the real image is excluding the padding.

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
    print('Number of detections: ',N)
    print('Window: ',window)
    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    print('boxes',boxes.shape,' ',boxes)
    class_ids = detections[:N, 4].astype(np.int32)
    print('Class_ids: ',class_ids.shape,' ',class_ids)
    scores = detections[:N, 5]
    print('Scores: ',scores.shape,' ',scores)
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]
    print('Masks: ',masks.shape)# masks)
    # Compute scale and shift to translate coordinates to image domain.
    print(image_shape[0])
    print(window[2] - window[0])
    h_scale = image_shape[0] / (window[2] - window[0])
    print('h_scale: ',h_scale)
    w_scale = image_shape[1] / (window[3] - window[1])
    print('w_scale: ',w_scale)
    scale = min(h_scale, w_scale)
    shift = window[:2]  # y, x
    print('shift: ',shift)
    scales = np.array([scale, scale, scale, scale])
    print('scales: ',scales)
    shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
    print('shifts: ',shifts)
    # Translate bounding boxes to image domain
    boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
    print('boxes: ',boxes.shape,' ',boxes)
    # Filter out detections with zero area. Often only happens in early
    # stages of training when the network weights are still a bit random.
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
        full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
        if full_masks else np.empty((0,) + masks.shape[1:3])

    return boxes, class_ids, scores, full_masks
pb_filepath = './pb_out_new.pb'
with tf.gfile.FastGFile(pb_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
print('Graph loaded.')
#testImage = "./flpoly_floor_xin/originals/data2.jpg"#image of the size defined in the config
testImage = "./151.jpg"
sess = tf.InteractiveSession()
image = imageio.imread(testImage)
print('Image loaded.')
images = [image]
print("Processing {} images".format(len(images)))
for im in images:
    modellib.log("image", im)
print('RGB image loaded and preprocessed.')
molded_images, image_metas, windows = mold_inputs(images)
print(molded_images.shape)
print('Images meta: ',image_metas)
image_shape = molded_images[0].shape

anchors = get_anchors(image_shape, inference_config)
# Duplicate across the batch dimension because Keras requires it
# TODO: can this be optimized to avoid duplicating the anchors?
inference_config.BATCH_SIZE = 1
image_anchors = np.broadcast_to(anchors, (inference_config.BATCH_SIZE,) + anchors.shape)
print('anchors shape is', image_anchors.shape, image_anchors.dtype)



img_ph = sess.graph.get_tensor_by_name('input_image:0')
print(img_ph)
img_meta_ph = sess.graph.get_tensor_by_name('input_image_meta:0')
print(img_meta_ph)
img_anchors_ph = sess.graph.get_tensor_by_name('input_anchors:0')
print(img_anchors_ph)



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
    print('Calculating results for image#',i)
    final_rois, final_class_ids, final_scores, final_masks =\
    unmold_detections(detections[i], mrcnn_mask[i],
                                    image.shape, windows[i])
    results.append({
        "rois": final_rois,
        "class_ids": final_class_ids,
        "scores": final_scores,
        "masks": final_masks,
    })
r = results[0]
print(results)
#print(r)
#print ("scoare", r['scores'])
#print (r['class_ids'][0])
#print (r['rois'][0])

print ("mask shape", r['masks'])


image = cv2.imread(testImage)
N = r['rois'].shape[0]  #find out how many different instances are there
try: #if no mask simply return image frame
    mask = r['masks']
    color = visualize.random_colors(N) #generates random instance colors
    color = color[0]
    image = visualize.apply_mask(masked_image, mask, color)
except Exception as e:
    print(e)
cv2.imshow('test', image)
cv2.waitKey(0)







#class_names = ["BG","nuclei"]
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], ax=get_ax())
print('Done')