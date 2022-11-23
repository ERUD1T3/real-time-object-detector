# where the model is defined
# source: https://github.com/xuannianz/keras-GaussianYOLOv3/blob/master/model.py

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.engine as KE
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Input
from tensorflow.keras.layers import Lambda, LeakyReLU, UpSampling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from utils import correct_boxes_and_scores_graph, yolo_loss, compose


class DetectionLayer(KE.Layer):
    def __init__(self,
                 anchors,
                 num_classes=20,
                 max_boxes_per_class_per_image=20,
                 score_threshold=.2,
                 iou_threshold=.5,
                 max_boxes_per_image=400,
                 **kwargs
                 ):
        super(DetectionLayer, self).__init__(**kwargs)
        self.anchors = anchors
        self.iou_threshold = iou_threshold
        self.max_boxes_per_class_per_image = max_boxes_per_class_per_image
        self.max_boxes_per_image = max_boxes_per_image
        self.num_classes = num_classes
        self.score_threshold = score_threshold

    def call(self, inputs, **kwargs):
        yolo_outputs = inputs[:-1]
        batch_image_shape = inputs[-1]
        num_output_layers = len(yolo_outputs)
        num_anchors_per_layer = len(self.anchors) // num_output_layers
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # tensor, (2, )
        input_shape = K.shape(yolo_outputs[0])[1:3] * 32
        grid_shapes = [K.shape(yolo_outputs[l])[1:3] for l in range(num_output_layers)]
        boxes_all_layers = []
        scores_all_layers = []
        for l in range(num_output_layers):
            yolo_output = yolo_outputs[l]
            grid_shape = grid_shapes[l]
            raw_y_pred = K.reshape(yolo_output,
                                   [-1, grid_shape[0], grid_shape[1], num_anchors_per_layer, self.num_classes + 9])
            boxes_this_layer, scores_this_layer = correct_boxes_and_scores_graph(raw_y_pred,
                                                                                 self.anchors[anchor_mask[l]],
                                                                                 self.num_classes,
                                                                                 input_shape,
                                                                                 batch_image_shape,
                                                                                 )
            boxes_all_layers.append(boxes_this_layer)
            scores_all_layers.append(scores_this_layer)

        # (b, total_num_anchors_all_layers, 4)
        boxes = K.concatenate(boxes_all_layers, axis=1)
        # (b, total_num_anchors_all_layers, num_classes)
        scores = K.concatenate(scores_all_layers, axis=1)
        mask = scores >= self.score_threshold
        max_boxes_per_class_per_image_tensor = K.constant(self.max_boxes_per_class_per_image, dtype='int32')
        max_boxes_per_image_tensor = K.constant(self.max_boxes_per_image, dtype='int32')

        def evaluate_batch_item(batch_item_boxes, batch_item_scores, batch_item_mask):
            boxes_per_class = []
            scores_per_class = []
            class_ids_per_class = []
            for c in range(self.num_classes):
                class_boxes = tf.boolean_mask(batch_item_boxes, batch_item_mask[:, c])
                # (num_keep_this_class_boxes, )
                class_scores = tf.boolean_mask(batch_item_scores[:, c], batch_item_mask[:, c])
                nms_keep_indices = tf.image.non_max_suppression(class_boxes,
                                                                class_scores,
                                                                max_boxes_per_class_per_image_tensor,
                                                                iou_threshold=self.iou_threshold)
                class_boxes = K.gather(class_boxes, nms_keep_indices)
                class_scores = K.gather(class_scores, nms_keep_indices)
                # (num_keep_this_class_boxes, )
                class_class_ids = K.ones_like(class_scores, 'float32') * c
                boxes_per_class.append(class_boxes)
                scores_per_class.append(class_scores)
                class_ids_per_class.append(class_class_ids)
            batch_item_boxes = K.concatenate(boxes_per_class, axis=0)
            batch_item_scores = K.concatenate(scores_per_class, axis=0)
            batch_item_scores = K.expand_dims(batch_item_scores, axis=-1)
            batch_item_class_ids = K.concatenate(class_ids_per_class, axis=0)
            batch_item_class_ids = K.expand_dims(batch_item_class_ids, axis=-1)
            # (num_keep_all_class_boxes, 6)
            batch_item_predictions = K.concatenate([batch_item_boxes,
                                                    batch_item_scores,
                                                    batch_item_class_ids], axis=-1)
            batch_item_num_predictions = tf.shape(batch_item_boxes)[0]
            batch_item_num_predictions = tf.Print(batch_item_num_predictions, [batch_item_num_predictions], '\nbatch_item_num_predictions', summarize=1000)
            batch_item_num_pad = tf.maximum(max_boxes_per_image_tensor - batch_item_num_predictions, 0)
            padded_batch_item_predictions = tf.pad(tensor=batch_item_predictions,
                                                   paddings=[
                                                       [0, batch_item_num_pad],
                                                       [0, 0]],
                                                   mode='CONSTANT',
                                                   constant_values=0.0)
            return padded_batch_item_predictions

        predictions = tf.map_fn(lambda x: evaluate_batch_item(x[0], x[1], x[2]),
                                elems=(boxes, scores, mask),
                                dtype=tf.float32)

        predictions = tf.reshape(predictions, (-1, self.max_boxes_per_image, 6))
        return predictions

    def compute_output_shape(self, input_shape):
        return None, self.max_boxes_per_image, 6


def darknet_conv2d(*args, **kwargs):
    """
    Wrapper to set Darknet parameters for Convolution2D.
    """
    darknet_conv_kwargs = dict({'kernel_regularizer': l2(5e-4)})
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def darknet_conv2d_bn_leaky(*args, **kwargs):
    """
    Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        darknet_conv2d(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    """
    A series of resblocks starting with a downsampling Convolution2D
    """
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = darknet_conv2d_bn_leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            darknet_conv2d_bn_leaky(num_filters // 2, (1, 1)),
            darknet_conv2d_bn_leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    """
    Darknet body having 52 Convolution2D layers
    1 + (1 + 1 * 2) + (1 + 2 * 2) + (1 + 8 * 2) + (1 + 8 * 2) + (1 + 4 * 2) = 1 + 3 + 5 + 17 + 17 + 9 = 52
    """
    # (416, 416)
    x = darknet_conv2d_bn_leaky(32, (3, 3))(x)
    # (208, 208)
    x = resblock_body(x, 64, 1)
    # (104, 104)
    x = resblock_body(x, 128, 2)
    # (52, 52)
    x = resblock_body(x, 256, 8)
    # (26, 26)
    x = resblock_body(x, 512, 8)
    # (13, 13)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    """
    6 conv2d_bn_leaky layers followed by a conv2d layer
    """
    x = compose(darknet_conv2d_bn_leaky(num_filters, (1, 1)),
                darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d_bn_leaky(num_filters, (1, 1)),
                darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d_bn_leaky(num_filters, (1, 1)))(x)
    y = compose(darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(anchors, num_classes=20, score_threshold=0.01):
    """
    Create YOLO_V3 model CNN body in Keras.

    Args:
        anchors:
        num_classes:
        score_threshold:

    Returns:

    """
    num_anchors = len(anchors)
    num_anchors_per_layer = num_anchors // 3
    image_input = Input(shape=(None, None, 3), name='image_input')
    fm_13_input = Input(shape=(None, None, num_anchors_per_layer, num_classes + 5), name='fm_13_input')
    fm_26_input = Input(shape=(None, None, num_anchors_per_layer, num_classes + 5), name='fm_26_input')
    fm_52_input = Input(shape=(None, None, num_anchors_per_layer, num_classes + 5), name='fm_52_input')
    image_shape_input = Input(shape=(2,), name='image_shape_input')
    darknet = Model([image_input], darknet_body(image_input))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors_per_layer * (num_classes + 9))
    x = compose(darknet_conv2d_bn_leaky(256, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors_per_layer * (num_classes + 9))
    x = compose(darknet_conv2d_bn_leaky(128, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors_per_layer * (num_classes + 9))

    loss = Lambda(yolo_loss,
                  output_shape=(1,),
                  name='yolo_loss',
                  arguments={'anchors': anchors,
                             'num_anchors_per_layer': num_anchors_per_layer,
                             'num_classes': num_classes,
                             'ignore_thresh': 0.5})(
        [y1, y2, y3, fm_13_input, fm_26_input, fm_52_input])
    training_model = Model([image_input, fm_13_input, fm_26_input, fm_52_input], loss, name='yolo')
    detections = DetectionLayer(anchors, num_classes=num_classes, score_threshold=score_threshold, name='yolo_detection')(
        [y1, y2, y3, image_shape_input])
    prediction_model = Model([image_input, image_shape_input], detections, name='yolo')
    return training_model, prediction_model