import tensorflow as tf
from tensorflow.keras import layers, initializers, activations, Input
import tensorflow_addons as tfa


class Encoder(layers.Layer):

    def __init__(self, name):
        super(Encoder, self).__init__()
        self.model = [
            layers.Conv2D(filters=64, 
                          kernel_size=(7, 7), 
                          padding="same", 
                          name="{}_Conv2D1".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                   name="{}_IN1".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu1".format(name)),
            layers.Conv2D(filters=128, 
                          kernel_size=(3, 3), 
                          strides=(2, 2), 
                          padding="same", 
                          name="{}_Conv2D2".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                   name="{}_IN2".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu2".format(name)),
            layers.Conv2D(filters=256, 
                          kernel_size=(3, 3), 
                          strides=(2, 2), 
                          padding="same", 
                          name="{}_Conv2D3".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                   name="{}_IN3".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu3".format(name)),
            ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class ResBlock(layers.Layer):

    def __init__(self, name):
        super(ResBlock, self).__init__()
        self.model = [
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", name="{}_Conv2D1".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                   name="{}_IN1".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu1".format(name)),
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", name="{}_Conv2D2".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                   name="{}_IN2".format(name)),
            layers.Activation(activations.relu, name="{}_ReLuEnd".format(name))
            ]

    def call(self, x):
        y = x
        for layer in self.model:
            if "ReLuEnd" in layer.name:
                y = layer(x+y)
            else:
                y = layer(y)
        return y


class AutoEncoder(layers.Layer):

    def __init__(self, name):
        super(AutoEncoder, self).__init__()
        self.model = [
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="{}_Conv2D1".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                   name="{}_IN1".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu1".format(name)),
            layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same", name="{}_Conv2D2".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                   name="{}_IN2".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu2".format(name)),
            layers.Conv2DTranspose(filters=64, 
                                   kernel_size=(3, 3), 
                                   padding="same", 
                                   name="{}_DeConv2D1".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                   name="{}_IN3".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu3".format(name)),
            layers.Conv2DTranspose(filters=256, 
                                   kernel_size=(3, 3),  
                                   padding="same", 
                                   name="{}_DeConv2D2".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                   name="{}_IN4".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu4".format(name)),
            ]

    def call(self, x):
        y = x
        for layer in self.model:
            y = layer(y)
        return y


class Decoder(layers.Layer):

    def __init__(self, name):
        super(Decoder, self).__init__()
        self.model = [
            layers.Conv2DTranspose(filters=128, 
                                   kernel_size=(3, 3), 
                                   strides=(2, 2),
                                   padding="same", 
                                   name="{}_DeConv2D1".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                   name="{}_IN1".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu1".format(name)),
            layers.Conv2DTranspose(filters=64, 
                                   kernel_size=(3, 3), 
                                   strides=(2, 2), 
                                   padding="same", 
                                   name="{}_DeConv2D2".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                   name="{}_IN2".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu2".format(name)),
            layers.Conv2D(filters=1, 
                          kernel_size=(7, 7),  
                          padding="same", 
                          name="{}_Conv2D1".format(name)),
            layers.Activation(activations.sigmoid, name="{}_sigmoid1".format(name))
            ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class LSTMConv2DBlock(layers.Layer):
    
    def __init__(self, name):
        super(LSTMConv2DBlock, self).__init__()
        self.model = [
            layers.ConvLSTM2D(filters=256,
                              kernel_size=(5, 5),
                              padding="same",
                              strides=(1, 1), 
                              return_sequences=True,
                              return_state=True,
                              activation='relu',
                              name="{}_LSTM1".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                   name="{}_IN1".format(name)),           
            layers.ConvLSTM2D(filters=256,
                              kernel_size=(5, 5),
                              padding="same",
                              strides=(1, 1), 
                              return_sequences=True,
                              return_state=True,
                              activation='relu', 
                              name="{}_LSTM2".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                   name="{}_IN2".format(name)),
        ]

    def call(self, x, initial_state=None):
        encoder_state = None
        batch = x.shape[0]
        for layer in self.model:
            if isinstance(layer, layers.ConvLSTM2D):
                x, m_state, c_state = layer(x, initial_state=initial_state)
                encoder_state = [m_state, c_state]
            else:
                x = layer(x)
        x = tf.reshape(x, (-1, 16, 21, 256))
        return x, encoder_state


if __name__=="__main__":
    model = Encoder(name="Generator")
    output = model(Input(shape=(4, 16, 84, 1)))
    # (None, 4, 4, 21, 256)
    print(output.shape)

    model = ResBlock(name="Block1")
    output = model(Input(shape=(4, 4, 21, 256)))
    # (None, 4, 4, 21, 256)
    print(output.shape)

    model = Decoder(name="Decoder")
    output = model(Input(shape=(16, 21, 256)))
    # (None, 4, 4, 21, 256)
    print(output.shape)
