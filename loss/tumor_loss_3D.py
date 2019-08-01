import keras.backend as K
import tensorflow as tf


def weighted_crossentropy(y_true, y_pred):
    y_pred = y_pred[:,:,:,1:7,:]
    y_true = y_true[:,:,:,1:7,:]
    y_pred_f = K.reshape(y_pred, (-1,3))
    y_true_f = K.reshape(y_true, (-1,))

    y_pred_f = K.log(tf.clip_by_value(y_pred_f, 1e-10, 1.0))

    neg = K.equal(y_true_f, K.zeros_like(y_true_f))
    neg_calculoss = tf.gather(y_pred_f[:,0], tf.where(neg))

    pos1 = K.equal(y_true_f, K.ones_like(y_true_f))
    pos1_calculoss = tf.gather(y_pred_f[:,1], tf.where(pos1))

    pos2 = K.equal(y_true_f, 2*K.ones_like(y_true_f))
    pos2_calculoss = tf.gather(y_pred_f[:,2], tf.where(pos2))

    # loss = -K.mean(tf.concat([0.78*neg_calculoss, 0.65*pos1_calculoss, 8.57*pos2_calculoss], 0))
    loss = -K.mean(tf.concat([1*neg_calculoss, 1*pos1_calculoss, 3*pos2_calculoss], 0))

    return loss

def Dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_pred = y_pred[:,:,:,1:7,:]
    y_true = y_true[:,:,:,1:7,:]
    y_pred_f = K.reshape(y_pred, (-1,3))
    y_true_f = K.reshape(y_true, (-1,))

    neg = tf.cast(K.equal(y_true_f, K.zeros_like(y_true_f)), 'float32')
    neg_inter = K.sum(neg * y_pred_f[:,0])
    neg_calculoss = (2* neg_inter + smooth) / (K.sum(neg) + K.sum(y_pred_f[:,0]) + smooth)
    neg_calculoss = 1-neg_calculoss
    pos1 = tf.cast(K.equal(y_true_f, K.ones_like(y_true_f)), 'float32')
    pos1_inter = K.sum(pos1 * y_pred_f[:,1])
    pos1_calculoss = (2* pos1_inter + smooth) / (K.sum(pos1) + K.sum(y_pred_f[:,1]) + smooth)
    pos1_calculoss = 1-pos1_calculoss
    pos2 = tf.cast(K.equal(y_true_f, 2*K.ones_like(y_true_f)), 'float32')
    pos2_inter = K.sum(pos2 * y_pred_f[:,2])
    pos2_calculoss = (2* pos2_inter + smooth) / (K.sum(pos2) + K.sum(y_pred_f[:,2]) + smooth)
    pos2_calculoss = 1-pos2_calculoss

    loss = 1*neg_calculoss + 0.7*pos1_calculoss + 8.57*pos2_calculoss

    return loss

def F_score_loss(y_true, y_pred):
    smooth = 1e-5
    b = 2
    y_pred = y_pred[:,:,:,1:7,:]
    y_true = y_true[:,:,:,1:7,:]
    y_pred_f = K.reshape(y_pred, (-1,3))
    y_true_f = K.reshape(y_true, (-1,))
    
    neg = tf.cast(K.equal(y_true_f, K.zeros_like(y_true_f)), 'float32')
    neg_inter = K.sum(neg * y_pred_f[:,0])
    neg_p = (neg_inter+smooth)/(K.sum(y_pred_f[:,0])+smooth)
    neg_r = (neg_inter+smooth)/(K.sum(neg)+smooth)
    neg_calculoss = (1+b**2)*neg_p*neg_r/((b**2)*neg_p+neg_r)
    neg_calculoss = 1-neg_calculoss
    
    pos1 = tf.cast(K.equal(y_true_f, K.ones_like(y_true_f)), 'float32')
    pos1_inter = K.sum(pos1 * y_pred_f[:,1])
    pos1_p = (pos1_inter+smooth)/(K.sum(y_pred_f[:,1])+smooth)
    pos1_r = (pos1_inter+smooth)/(K.sum(pos1)+smooth)
    pos1_calculoss = (1+b**2)*pos1_p*pos1_r/((b**2)*pos1_p+pos1_r)
    pos1_calculoss = 1-pos1_calculoss
    
    pos2 = tf.cast(K.equal(y_true_f, 2*K.ones_like(y_true_f)), 'float32')
    pos2_inter = K.sum(pos2 * y_pred_f[:,2])
    pos2_p = (pos2_inter+smooth)/(K.sum(y_pred_f[:,2])+smooth)
    pos2_r = (pos2_inter+smooth)/(K.sum(pos2)+smooth)
    pos2_calculoss = (1+b**2)*pos2_p*pos2_r/((b**2)*pos2_p+pos2_r)
    pos2_calculoss = 1-pos2_calculoss

    loss = 1*neg_calculoss + 0.7*pos1_calculoss + 8.57*pos2_calculoss

    return loss

def Dice(y_true, y_pred):

    smooth = 1e-5
    y_pred = y_pred[:,:,:,1:7,:]
    y_true = y_true[:,:,:,1:7,:]
    
    y_true_f = K.reshape(y_true, (-1,))
    y_true_f = tf.cast(K.equal(y_true_f, 2*K.ones_like(y_true_f)), 'float32')
    
    y_pred_f = K.reshape(y_pred, (-1,3))
    y_pred_f = y_pred_f[:,2]
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2* intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def recall(y_true, y_pred):
    smooth = 1e-5
    y_pred = y_pred[:,:,:,1:7,:]
    y_true = y_true[:,:,:,1:7,:]
    
    y_true_f = K.reshape(y_true, (-1,))
    y_true_f = tf.cast(K.equal(y_true_f, 2*K.ones_like(y_true_f)), 'float32')
    
    y_pred_f = K.reshape(y_pred, (-1,3))
    y_pred_f = y_pred_f[:,2]
    
    inter = K.sum(y_pred_f*y_true_f)
    recall = (inter+smooth)/(K.sum(y_true_f)+smooth)
    return recall
