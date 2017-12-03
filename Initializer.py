import tensorflow as tf
import numpy as np
from keras import backend as K

def tao_method(shape, dtype=None):
    # return K.random_normal(shape, dtype=dtype)
    return tf.convert_to_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))

