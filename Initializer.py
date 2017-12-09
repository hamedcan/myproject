import tensorflow as tf
import numpy as np
from Init_data import InitData

def tao_method(shape=(3,3,3), dtype=None):
    data = InitData.__call__()
    return tf.convert_to_tensor(np.array(data.get()))

