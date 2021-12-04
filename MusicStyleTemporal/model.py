import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential, Input, layers, initializers, activations, optimizers, losses
from module import Encoder, ResBlock, Decoder, AutoEncoder, LSTMConv2DBlock, TransformerBlock, PositionEncoder


class LSTMGenerator(Model):

    def __init__(self, args, name):
        super(LSTMGenerator, self).__init__()
    
        # optimizer
        if args:
            self.optimizer = optimizers.Adam(learning_rate=args.lrg, beta_1=args.beta1)

        # architexture 
        
        nameS = "{}_StyleEncoder".format(name)
        self.styleEncoder = [Encoder(nameS),
                             LSTMConv2DBlock("{}1".format(nameS), last_Norm=False, cnn=False),
                             ]
        self.autoEncoder = AutoEncoder("AutoEncoder")
       
        # architexture
        nameA = "{}_AudioEncoder".format(name)
        self.audioEncoder = [Encoder(nameA),
                             LSTMConv2DBlock("{}1".format(nameA), last_Norm=True, cnn=True),
                             ResBlock("{}1".format(nameA)),
                             ResBlock("{}2".format(nameA)),
                             ResBlock("{}3".format(nameA)),
                             ResBlock("{}4".format(nameA)),
                             ResBlock("{}5".format(nameA)),
                             ResBlock("{}6".format(nameA)),
                             ResBlock("{}7".format(nameA)),
                             ResBlock("{}8".format(nameA)),
                             ResBlock("{}9".format(nameA)),
                             ResBlock("{}10".format(nameA)),
                             Decoder(name)
                             ]
     
    def call(self, audio, style):
        batch = audio.shape[0]
        style = tf.reshape(style, (-1, 64, 1, 84, 1))
        audio = tf.reshape(audio, (-1, 64, 1, 84, 1))
        
        for layer in self.styleEncoder:
            if isinstance(layer, LSTMConv2DBlock):
                style, encoder_state = layer(style)
            else:
                style = layer(style) 
        # encoder_state[0]: (batch, 1, 21, 128)
        # latent code: (batch, 1, 3, 128)
        encoder_state[0], encoder_state[1] = self.autoEncoder(encoder_state[0]), self.autoEncoder(encoder_state[1])
        for layer in self.audioEncoder:
            if isinstance(layer, LSTMConv2DBlock):
                audio, _ = layer(audio, initial_state=encoder_state)
            else:
                audio = layer(audio)
        return audio
    
    #@staticmethod
    def loss_fn(self, generation, reconstruction, original):
          G_loss = tf.reduce_mean(tf.square(generation-tf.ones_like(generation)))
          Cycle_loss = tf.reduce_mean(tf.abs(original-reconstruction)) 
          return G_loss+Cycle_loss 
 

class Discriminator(Model):

    def __init__(self, args, name):
        super(Discriminator, self).__init__()
    
        # optimizer
        if args:
            self.optimizer = optimizers.Adam(learning_rate=args.lrd, beta_1=args.beta1)

        # architexture
        self.architecture = Sequential([layers.Conv2D(
                                            filters=64, 
                                            kernel_size=(7, 7), 
                                            strides=(2, 2),
                                            padding="same",
                                            name="{}_Conv2D1".format(name)),
                                        layers.LeakyReLU(alpha=0.2, name="{}_LReLu1".format(name)),
                                        layers.Conv2D(
                                            filters=256, 
                                            kernel_size=(7, 7), 
                                            strides=(2, 2),
                                            padding="same",
                                            name="{}_Conv2D2".format(name)),
                                        tfa.layers.InstanceNormalization(
                                            axis=3,
                                            center=True,
                                            scale=True,
                                            beta_initializer=initializers.Constant(value=0),
                                            gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                            name="{}_IN1".format(name)),
                                        layers.LeakyReLU(alpha=0.2, name="{}_LReLu2".format(name)),
                                        layers.Conv2D(
                                            filters=1, 
                                            kernel_size=(7, 7),
                                            padding="same",
                                            name="{}_Conv2D3".format(name)),
                                        ])
    
    def call(self, x):
        x = self.architecture(x)
        return x

    #@staticmethod
    def loss_fn(self, original, generation):#, mix_original, mix_generation):
        D_loss = tf.reduce_mean(tf.square(original-tf.ones_like(original)))
        D_loss += tf.reduce_mean(tf.square(generation-tf.zeros_like(generation)))
        return 0.5*D_loss


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
    
class TransformerGenerator(Model):

    def __init__(self, args, name):
        super(TransformerGenerator, self).__init__()

        # optimizer
        if args:
            self.optimizer = optimizers.Adam(learning_rate=args.lrg, beta_1=args.beta1)

        # architexture
        '''self.architecture = Sequential([layers.Embedding()
                                        Position_Encoding_Layer(),
                                        TransformerBlock(),
                                        layers.Dense(),
                                        TransformerBlock(),
                                        layers.Dense(),
                                        Decoder()
                                        ])'''
        # I have not added dimensions for anything yet
        self.emb_sz = 84
        self.architecture = [# Embedding??
                            PositionEncoder(64, self.emb_sz),
                            TransformerBlock(self.emb_sz), # Self Attention
                            layers.Dense(84)
                            ]

    def call(self, x, y):
        x = tf.reshape(x, (-1,64,84))
        for layer in self.architecture:
            x = layer(x)
        x = tf.reshape(x, (-1,64,84,1))
        return x

    #@staticmethod
    def loss_fn(self, generation, reconstruction, original):
          G_loss = tf.reduce_mean(tf.square(generation-tf.ones_like(generation)))
          Cycle_loss = tf.reduce_mean(tf.abs(original-reconstruction))
          return G_loss+10*Cycle_loss

if __name__=="__main__":
    # comment anything related to args in model
    model = Generator(args=None, name="Generator")
    output = model(Input(shape=(64, 84, 1)))
    # (None, 64, 84, 1)
    print(output.shape)
    model.architecture.summary()

    # comment anything related to args in model
    model = Discriminator(args=None, name="Discriminator")
    output = model(Input(shape=(64, 84, 1)))
    # (None, 256, 256, 1)
    print(output.shape)
    model.architecture.summary()
