import keras.backend as K
import tensorflow as tf

def weighted_crossentropy_2ddense(NUM_CLASS=1):

    def loss(label, pred):
        if NUM_CLASS==1:
            y_true_f = K.flatten(label)
            y_pred_f = K.flatten(pred)
            y_pred_f = K.log(tf.clip_by_value(y_pred_f, 1e-10, 1.0))
            neg = 1-y_true_f
            neg_pred = 1-y_pred_f
        else:
            y_true_f = K.reshape(label, (-1,))
            y_pred_f = K.reshape(pred, (-1,2))
            y_pred_f = K.log(tf.clip_by_value(y_pred_f, 1e-10, 1.0))
            neg=1-y_true_f
            neg_pred = y_pred_f[:,0]
            y_pred_f = y_pred_f[:,1]

        neg_calculoss = tf.gather(neg_pred, tf.where(neg))

        pos1 = K.equal(y_true_f, K.ones_like(y_true_f))
        pos1_calculoss = tf.gather(y_pred_f, tf.where(pos1))

        return -K.mean(tf.concat([0.8*neg_calculoss, 5.5*pos1_calculoss], 0))
    return loss

def Dice_loss(NUM_CLASS=1):
    
    def loss(label, pred):
        smooth = 1e-5
        if NUM_CLASS==1:
            y_true_f = K.flatten(label)
            y_pred_f = K.flatten(pred)
            neg = 1-y_true_f
            neg_pred = 1-y_pred_f
        else:
            y_true_f = K.reshape(label, (-1,))
            y_pred_f = K.reshape(pred, (-1,2))
            neg=1-y_true_f
            neg_pred = y_pred_f[:,0]
            y_pred_f = y_pred_f[:,1]

        neg_inter = K.sum(neg * neg_pred)
        neg_calculoss = (2* neg_inter + smooth) / (K.sum(neg) + K.sum(neg_pred) + smooth)
        neg_calculoss = 1-neg_calculoss

        pos1 = tf.cast(K.equal(y_true_f, K.ones_like(y_true_f)), 'float32')
        pos1_inter = K.sum(pos1 * y_pred_f)
        pos1_calculoss = (2* pos1_inter + smooth) / (K.sum(pos1) + K.sum(y_pred_f) + smooth)
        pos1_calculoss = 1-pos1_calculoss

        return 0.78*neg_calculoss + 5.5*pos1_calculoss

    return loss

def F_score_loss(NUM_CLASS=1):
    
    def loss(label, pred):
        smooth = 1e-5
        b = 3
        if NUM_CLASS==1:
            y_true_f = K.flatten(label)
            y_pred_f = K.flatten(pred)
            neg = 1-y_true_f
            neg_pred = 1-y_pred_f
        else:
            y_true_f = K.reshape(label, (-1,))
            y_pred_f = K.reshape(pred, (-1,2))
            neg=1-y_true_f
            neg_pred = y_pred_f[:,0]
            y_pred_f = y_pred_f[:,1]
        
        neg_inter = K.sum(neg * neg_pred)
        neg_p = (neg_inter+smooth)/(K.sum(neg_pred)+smooth)
        neg_r = (neg_inter+smooth)/(K.sum(neg)+smooth)
        neg_calculoss = (1+b**2)*neg_p*neg_r/((b**2)*neg_p+neg_r)
        neg_calculoss = 1-neg_calculoss
    
        pos1 = tf.cast(K.equal(y_true_f, K.ones_like(y_true_f)), 'float32')
        pos1_inter = K.sum(pos1 * y_pred_f)
        pos1_p = (pos1_inter+smooth)/(K.sum(y_pred_f)+smooth)
        pos1_r = (pos1_inter+smooth)/(K.sum(pos1)+smooth)
        pos1_calculoss = (1+b**2)*pos1_p*pos1_r/((b**2)*pos1_p+pos1_r)
        pos1_calculoss = 1-pos1_calculoss

        return 0.5*neg_calculoss + 2*pos1_calculoss
    return loss

def Dice_1(NUM_CLASS=1):

    def Dice(label, pred):
        smooth = 1e-5
        if NUM_CLASS==1:
            y_true_f = K.flatten(label)
            y_pred_f = K.flatten(pred)
        else:
            y_true_f = K.reshape(label, (-1,))
            y_pred_f = K.reshape(pred, (-1,2))
            y_pred_f = y_pred_f[:,1]            
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2* intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice
    return Dice

def recall_1(NUM_CLASS=1):
    def recall(label, pred):

        smooth = 1e-5
        if NUM_CLASS==1:
            y_true_f = K.flatten(label)
            y_pred_f = K.flatten(pred)
        else:
            y_true_f = K.reshape(label, (-1,))
            y_pred_f = K.reshape(pred, (-1,2))
            y_pred_f = y_pred_f[:,1]   
        inter = K.sum(y_pred_f*y_true_f)
        recall = (inter+smooth)/(K.sum(y_true_f)+smooth)
        return recall
    return recall

