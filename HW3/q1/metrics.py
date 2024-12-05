import tensorflow as tf
from tensorflow.keras import backend as K

def iou_metric(y_true, y_pred, epsilon=1e-8):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection) / (union + epsilon)

def dice_coefficient(y_true, y_pred, epsilon=1e-8):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + epsilon)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)