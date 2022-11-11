import tensorflow as tf


class ClassWeightMult(tf.keras.layers.Layer):
    def __init__(self, class_weight):
        super().__init__()
        self.class_weight = class_weight


    def call(self, inputs):
        return inputs * self.class_weight


    def get_config(self):
        config = super().get_config()
        config.update({
            "class_weight": self.class_weight,
        })
        return config
