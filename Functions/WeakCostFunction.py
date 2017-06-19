# for custom metrics
import keras.backend as K

REGULARIZATION = 1e-8

# Function takes in two Keras tensors
# y_true is NOT the standard label for a classifier, but rather the
#   percentage of the total data set which are in class 1
# y_pred is the prediction for each point (in the range 0-1)
# the output is the cost
def WeakSupervision(y_true, y_pred):
    return K.abs((K.mean(y_pred, axis=-1) - K.mean(y_true,axis=-1)))

# Function takes in two Keras tensors
# y_true is NOT the standard label for a classifier, but rather the
#   percentage of the total data set which are in class 1
# y_pred is the prediction for each point (in the range 0-1)
# the output is the cost
def WeakSupervision_v2(y_true, y_pred):
    loss1 = K.mean(y_pred, axis=-1) - K.mean(y_true,axis=-1)
    constrib = REGULARIZATION * K.std(y_pred)
    loss1 = K.square(loss1) - constrib
    
    loss2 = (1.0 - K.mean(y_pred, axis=-1)) - K.mean(y_true,axis=-1)
    loss2 = K.square(loss2) - constrib
    
    #loss = K.switch(K.less(loss1,loss2),loss1,loss2)
    loss = K.minimum(loss1,loss2)
    return loss