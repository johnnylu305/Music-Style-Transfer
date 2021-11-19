import tensorflow as tf
from tensorflow.keras import layers, initializers, activations, Input, Model, Sequential, losses
import tensorflow_addons as tfa


class Classifier(Model):

    def __init__(self, args, name):
        super(Classifier, self).__init__()
        self.model = Sequential([
            layers.Conv2D(filters=64, 
                          kernel_size=(1, 12), 
                          strides=(1, 12),
                          padding="same", 
                          name="{}_Conv2D1".format(name)),
            layers.LeakyReLU(alpha=0.2, name="{}_LReLu1".format(name)),
            layers.Conv2D(filters=128, 
                          kernel_size=(4, 1), 
                          strides=(4, 1), 
                          padding="same", 
                          name="{}_Conv2D2".format(name)),
            tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                   name="{}_IN1".format(name)),
            layers.LeakyReLU(alpha=0.2, name="{}_LReLu2".format(name)),           
            layers.Conv2D(filters=256, 
                          kernel_size=(2, 1), 
                          strides=(2, 1), 
                          padding="same", 
                          name="{}_Conv2D3".format(name)),
            tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                   name="{}_IN3".format(name)),
            layers.LeakyReLU(alpha=0.2, name="{}_LReLu3".format(name)),     
            layers.Conv2D(filters=512, 
                          kernel_size=(8, 1), 
                          strides=(8, 1), 
                          padding="same", 
                          name="{}_Conv2D4".format(name)),
            tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                   name="{}_IN4".format(name)),
            layers.LeakyReLU(alpha=0.2, name="{}_LReLu4".format(name)),     
            layers.Conv2D(filters=1, 
                          kernel_size=(1, 7),
                          strides=(1, 7),
                          padding="same", 
                          name="{}_Conv2D5".format(name),
                          activation='sigmoid'),
            layers.Reshape(target_shape=[1])
            ])

    def call(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        # labels: (batch)
        # predictions: (batch, 1)
        return losses.BinaryCrossentropy(from_logits=False)(labels, predictions)


if __name__=="__main__":
    model = Classifier(name="Classifier")
    output = model(Input(shape=(64, 84, 1)))
    # (None, 1)
    print(output.shape)
    model.model.summary()
