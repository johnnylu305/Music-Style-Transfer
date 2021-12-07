import tensorflow as tf
from tensorflow.keras import layers, initializers, activations, Input
import tensorflow_addons as tfa


class Encoder(layers.Layer):

    def __init__(self, name):
        super(Encoder, self).__init__()
        self.model = [
            layers.Conv2D(filters=32, 
                          kernel_size=(1, 7), 
                          padding="same", 
                          name="{}_Conv2D1".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                   name="{}_IN1".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu1".format(name)),
            layers.Conv2D(filters=64, 
                          kernel_size=(1, 3), 
                          strides=(1, 2), 
                          padding="same", 
                          name="{}_Conv2D2".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                   name="{}_IN2".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu2".format(name)),
            layers.Conv2D(filters=128, 
                          kernel_size=(1, 3), 
                          strides=(1, 2), 
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
            layers.Conv2D(filters=128, 
                          kernel_size=(3, 3), 
                          strides=(1, 3), 
                          padding="same", 
                          name="{}_Conv2D1".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                   name="{}_IN1".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu1".format(name)),
            layers.Conv2DTranspose(filters=128, 
                                   kernel_size=(3, 3),
                                   strides=(1, 3),
                                   padding="same", 
                                   name="{}_DeConv2D1".format(name)),
            tfa.layers.InstanceNormalization(axis=-1, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer=initializers.Constant(value=0),
                                   gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                   name="{}_IN2".format(name)),
            layers.Activation(activations.relu, name="{}_ReLu2".format(name))
            ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class Decoder(layers.Layer):

    def __init__(self, name):
        super(Decoder, self).__init__()
        self.model = [
            layers.Conv2DTranspose(filters=128, 
                                   kernel_size=(3, 3), 
                                   strides=(1, 2),
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
                                   strides=(1, 2), 
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
    
    def __init__(self, name, last_Norm, cnn):
        super(LSTMConv2DBlock, self).__init__()
        self.model = [
            layers.ConvLSTM2D(filters=128,
                              kernel_size=(1, 3),
                              padding="same",
                              strides=(1, 1), 
                              return_sequences=True,
                              return_state=True,
                              name="{}_LSTM1".format(name))]
        if last_Norm:
            self.model.append(tfa.layers.InstanceNormalization(
                                axis=-1, 
                                center=True, 
                                scale=True,
                                beta_initializer=initializers.Constant(value=0),
                                gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                name="{}_IN1".format(name)))
        if cnn:
            self.model += [layers.Conv2D(
                                filters=256, 
                                kernel_size=(3, 3), 
                                padding="same", 
                                name="{}_Conv2D1".format(name)),
                           tfa.layers.InstanceNormalization(
                                axis=-1, 
                                center=True, 
                                scale=True,
                                beta_initializer=initializers.Constant(value=0),
                                gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                name="{}_INEnd".format(name)),
                           layers.Activation(activations.relu, name="{}_ReLu1".format(name))]

    def call(self, x, initial_state=None):
        encoder_state = None
        for layer in self.model:
            if isinstance(layer, layers.ConvLSTM2D):
                x, m_state, c_state = layer(x, initial_state=initial_state)
                encoder_state = [m_state, c_state]
                x = tf.reshape(x, (-1, 64, 21, 128))
            else:
                x = layer(x)
        return x, encoder_state
   

class TransformerBlock(layers.Layer):
    def __init__(self, emb_sz):
        super(TransformerBlock, self).__init__()

        self.heads = 3

        self.attention = tfa.layers.MultiHeadAttention(num_heads=self.heads, head_size=int(emb_sz/2))
        self.ff_layer = tf.keras.Sequential([layers.Dense(emb_sz, activation="relu"),
                                    layers.Dense(emb_sz)
                                    ])
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs, training):
        attention_output = self.attention([inputs, inputs, inputs])
        attention_output += inputs

        layer_norm1_output = self.layer_norm(attention_output)

        ff_output = self.ff_layer(layer_norm1_output)
        ff_output += layer_norm1_output

        layer_norm2_output = self.layer_norm(ff_output)

        return tf.nn.relu(layer_norm2_output)

# The one from transformer assignment
class PositionEncoder(layers.Layer):
	def __init__(self, sample_length, emb_sz):
		super(PositionEncoder, self).__init__()
		self.positional_embeddings = self.add_weight("pos_embed",shape=[sample_length, emb_sz])

	@tf.function
	def call(self, x):
		return x+self.positional_embeddings

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
