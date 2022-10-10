import tensorflow as tf


class Classifier(tf.keras.Model):
    def __init__(self, classes, data_format='channels_last'):
        super(Classifier, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)
        self.flt = tf.keras.layers.Flatten()
        self.softmax = tf.keras.layers.Softmax()
        self.den1 = tf.keras.layers.Dense(classes)

    def call(self, inputs):
        x = self.pool(inputs)
        x = self.flt(x)
        x = self.den1(x)
        x = self.softmax(x)
        return x
