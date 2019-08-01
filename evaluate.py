import numpy as np
def Dice(label, pred):
    import numpy as np
    smooth = 1e-5
    label = label.flatten()
    pred = pred.flatten()
    inter = np.sum(label*pred)
    D = (2*inter+smooth)/(np.sum(pred)+np.sum(label)+smooth)
    return D

def recall(label, pred):
    import numpy as np
    smooth = 1e-5
    label = label.flatten()
    pred = pred.flatten()
    inter = np.sum(label*pred)
    recall = (inter+smooth)/(np.sum(label)+smooth)
    return recall

def precision(label, pred):
    import numpy as np
    smooth = 1e-5
    label = label.flatten()
    pred = pred.flatten()
    inter = np.sum(label*pred)
    precision = (inter+smooth)/(np.sum(pred)+smooth)
    return precision