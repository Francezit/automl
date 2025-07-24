from keras.layers import *
import tensorflow as _tf

class DropPath(Layer):
    """The Drop path in ConvNeXt

        Reference:
            https://github.com/rishigami/Swin-Transformer-TF/blob/main/swintransformer/model.py
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return self._drop_path(x, self.drop_prob, training)

    
    def _drop_path(self, inputs, drop_prob, is_training):
        if (not is_training) or (drop_prob == 0.):
            return inputs

        # Compute keep_prob
        keep_prob = 1.0 - drop_prob

        # Compute drop_connect tensor
        random_tensor = keep_prob
        shape = (_tf.shape(inputs)[0],) + (1,) * \
            (len(_tf.shape(inputs)) - 1)
        random_tensor += _tf.random.uniform(shape, dtype=inputs.dtype)
        binary_tensor = _tf.floor(random_tensor)
        output = _tf.math.divide(inputs, keep_prob) * binary_tensor
        return output

__all__ = [
    name
    for name in globals().keys()
    if not (name.startswith("_"))
]