import keras.backend as K
import tensorflow as tf
import numpy as np
from functools import reduce

def box_iou_graph(b1, b2):
    """
    Return iou tensor

    Args:
        b1 (tensor): (fh, fw, num_anchors_this_layer, 4)
        b2 (tensor): (num_gt_boxes, 4)

    Returns:
        iou (tensor): shape=(num_b1_boxes, num_b2_boxes)
    """
    # Expand dim to apply broadcasting.
    # (fh, fw, num_anchors_this_layer, 1, 4)
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    # (1, num_gt_boxes, 4)
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # (fh, fw, num_anchors_this_layer, num_b2_boxes, 2)
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    # (fh, fw, num_anchors_this_layer, num_b2_boxes)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # (fh, fw, num_anchors_this_layer, 1)
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    # (1, num_gt_boxes)
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


def correct_boxes_graph(y_pred_xy, y_pred_wh, input_shape, image_shape):
    """

    Args:
        y_pred_xy: (b, fh, fw, num_anchors_this_layer, 2)
        y_pred_wh: (b, fh, fw, num_anchors_this_layer, 2)
        input_shape: (b, 2), hw
        image_shape: (b, 2), hw

    Returns:
        boxes: (b, fh, fw, num_anchors_this_layer, 4), (y_min, x_min, y_max, x_max)

    """
    box_yx = y_pred_xy[..., ::-1]
    box_hw = y_pred_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        # y_min
        box_mins[..., 0:1],
        # x_min
        box_mins[..., 1:2],
        # y_max
        box_maxes[..., 0:1],
        # x_max
        box_maxes[..., 1:2]
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def correct_boxes_and_scores_graph(raw_y_pred, anchors, num_classes, input_shape, image_shape):
    """
    Args:
        raw_y_pred:
        anchors: (num_anchors_this_layer, 2)
        num_classes:
        input_shape: (2, ) hw
        image_shape: (batch_size, 2)

    Returns:
        boxes: (b, total_num_anchors_this_layer, 4), (y_min, x_min, y_max, x_max)
        boxes_scores: (b, total_num_anchors_this_layer, num_classes)

    """
    _, y_pred_box, _, _, y_pred_sigma, y_pred_confidence, y_pred_class_probs = y_pred_graph(raw_y_pred, anchors, input_shape)
    y_pred_xy = y_pred_box[..., :2]
    y_pred_wh = y_pred_box[..., 2:]
    # for batch predictions
    batch_size = K.shape(image_shape)[0]
    input_shape = K.expand_dims(input_shape, axis=0)
    input_shape = K.tile(input_shape, (batch_size, 1))
    elems = (y_pred_xy, y_pred_wh, input_shape, image_shape)
    boxes = tf.map_fn(lambda x: correct_boxes_graph(x[0], x[1], x[2], x[3]), elems=elems, dtype=tf.float32)
    box_scores = y_pred_confidence * y_pred_class_probs * (1 - tf.reduce_mean(y_pred_sigma, axis=-1, keep_dims=True))
    boxes = tf.reshape(boxes, [batch_size, -1, 4])
    box_scores = tf.reshape(box_scores, [batch_size, -1, num_classes])
    return boxes, box_scores


def y_pred_graph(raw_y_pred, anchors, input_shape):
    """
    Convert final layer features to bounding box parameters.

    Args:
        raw_y_pred: (b, fh, fw, num_anchors_per_layer, 2 + 2 + 1 + num_classes)
        anchors:
        input_shape:

    Returns:
        grid: (fh, fw, 1, 2)
        y_pred_box: (b, fh, fw, num_anchors_this_layer, 2 + 2)
        y_pred_delta_xy:
        y_pred_log_wh:
        y_pred_confidence: (b, fh, fw, num_anchors_this_layer, 1)
        y_pred_class_probs: (b, fh, fw, num_anchors_this_layer, num_classes)
    """
    num_anchors_this_layer = len(anchors)
    # Reshape to (batch, height, width, num_anchors, box_params)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors_this_layer, 2])
    grid_shape = K.shape(raw_y_pred)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y], axis=-1)
    grid = K.cast(grid, K.dtype(raw_y_pred))
    y_pred_xy = (K.sigmoid(raw_y_pred[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(raw_y_pred))
    y_pred_wh = K.exp(raw_y_pred[..., 2:4]) * (anchors_tensor / K.cast(input_shape[::-1], K.dtype(raw_y_pred)))
    # (batch_size, grid_height, grid_width, num_anchors_this_layer, 4)
    y_pred_box = K.concatenate([y_pred_xy, y_pred_wh])
    y_pred_delta_xy = K.sigmoid(raw_y_pred[..., :2])
    y_pred_log_wh = raw_y_pred[..., 2:4]
    y_pred_sigma = K.sigmoid(raw_y_pred[..., 4:8])
    y_pred_confidence = K.sigmoid(raw_y_pred[..., 8:9])
    y_pred_class_probs = K.sigmoid(raw_y_pred[..., 9:])

    return grid, y_pred_box, y_pred_delta_xy, y_pred_log_wh, y_pred_sigma, y_pred_confidence, y_pred_class_probs

def nll_loss(x, mu, sigma, sigma_const=0.3):
    pi = tf.constant(np.pi)
    Z = (2 * pi * (sigma + sigma_const) ** 2) ** 0.5
    probability_density = tf.exp(-0.5 * (x - mu) ** 2 / ((sigma + sigma_const) ** 2)) / Z
    nll = -tf.log(probability_density + 1e-7)
    return nll


def yolo_loss(args, anchors, num_anchors_per_layer, num_classes, ignore_thresh=.5, print_loss=True):
    """
    Return yolo_loss tensor
    Args:
        args (list): args[:num_output_layers] the output of yolo_body or tiny_yolo_body
            args[num_output_layers:] raw_y_true
        anchors (np.array): shape=(N, 2), wh
        num_anchors_per_layer (int):
        num_classes (int):
        ignore_thresh (float): the iou threshold whether to ignore object confidence loss
        print_loss:
    Returns:
        loss: tensor, shape=(1,)
    """
    num_output_layers = len(anchors) // num_anchors_per_layer
    yolo_outputs = args[:num_output_layers]
    raw_y_trues = args[num_output_layers:]
    anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(raw_y_trues[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(raw_y_trues[0])) for l in range(num_output_layers)]
    loss = 0
    batch_size = K.shape(yolo_outputs[0])[0]
    batch_size_f = K.cast(batch_size, K.dtype(yolo_outputs[0]))

    for l in range(num_output_layers):
        grid_shape = grid_shapes[l]
        yolo_output = yolo_outputs[l]
        raw_y_pred = K.reshape(yolo_output, [-1, grid_shape[0], grid_shape[1], num_anchors_per_layer, num_classes + 9])
        raw_y_true = raw_y_trues[l]
        anchor_mask = anchor_masks[l]
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, 1)
        object_mask = raw_y_true[..., 4:5]
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, num_classes)
        y_true_class_probs = raw_y_true[..., 5:]
        grid, y_pred_box, y_pred_delta_xy, y_pred_log_wh, y_pred_sigma, y_pred_confidence, y_pred_class_probs = \
            y_pred_graph(raw_y_pred, anchors[anchor_mask], input_shape)
        y_true_delta_xy = raw_y_true[..., :2] * grid_shapes[l][::-1] - grid
        y_true_log_wh = K.log(raw_y_true[..., 2:4] * input_shape[::-1] / anchors[anchor_mask])
        y_true_log_wh = K.switch(object_mask, y_true_log_wh, K.zeros_like(y_true_log_wh))
        box_loss_scale = 2 - raw_y_true[..., 2:3] * raw_y_true[..., 3:4]
        ignore_mask = tf.TensorArray(K.dtype(raw_y_trues[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask_):
            # (num_gt_boxes, 4)
            gt_box = tf.boolean_mask(raw_y_true[b, ..., 0:4], object_mask_bool[b, ..., 0])
            # (grid_height, grid_width, num_anchors_this_layer, num_gt_boxes)
            iou = box_iou_graph(y_pred_box[b], gt_box)
            # (grid_height, grid_width, num_anchors_this_layer)
            best_iou = K.max(iou, axis=-1)
            ignore_mask_ = ignore_mask_.write(b, K.cast(best_iou < ignore_thresh, K.dtype(gt_box)))
            return b + 1, ignore_mask_

        _, ignore_mask = tf.while_loop(lambda b, *largs: b < batch_size, loop_body, [0, ignore_mask])
        # (batch_size, grid_height, grid_width, num_anchors_this_layer)
        ignore_mask = ignore_mask.stack()
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, 1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        y_true = tf.concat([y_true_delta_xy, y_true_log_wh], axis=-1)
        y_pred_mu = tf.concat([y_pred_delta_xy, y_pred_log_wh], axis=-1)
        x_loss = nll_loss(y_true[..., 0:1], y_pred_mu[..., 0:1], y_pred_sigma[..., 0:1])
        x_loss = object_mask * box_loss_scale * x_loss
        y_loss = nll_loss(y_true[..., 1:2], y_pred_mu[..., 1:2], y_pred_sigma[..., 1:2])
        y_loss = object_mask * box_loss_scale * y_loss
        w_loss = nll_loss(y_true[..., 2:3], y_pred_mu[..., 2:3], y_pred_sigma[..., 2:3])
        w_loss = object_mask * box_loss_scale * w_loss
        h_loss = nll_loss(y_true[..., 3:4], y_pred_mu[..., 3:4], y_pred_sigma[..., 3:4])
        h_loss = object_mask * box_loss_scale * h_loss
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, y_pred_confidence) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, y_pred_confidence) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(y_true_class_probs, y_pred_class_probs)
        x_loss = K.sum(x_loss) / batch_size_f
        y_loss = K.sum(y_loss) / batch_size_f
        w_loss = K.sum(w_loss) / batch_size_f
        h_loss = K.sum(h_loss) / batch_size_f
        confidence_loss = K.sum(confidence_loss) / batch_size_f
        class_loss = K.sum(class_loss) / batch_size_f
        loss += x_loss + y_loss + w_loss + h_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, x_loss, y_loss, w_loss, h_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='\nloss: ')
    return loss

def compose(*funcs):
    """
    Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')