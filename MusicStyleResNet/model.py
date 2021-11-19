import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential, Input, layers, initializers, activations, optimizers
from module import Encoder, ResBlock, Decoder


class Generator(Model):

    def __init__(self, args, name):
        super(Generator, self).__init__()
    
        # optimizer
        if args:
            self.optimizer = optimizers.Adam(learning_rate=args.lr)

        # architexture
        self.architecture = Sequential([Encoder(name),
                                        ResBlock("{}1".format(name)),
                                        ResBlock("{}2".format(name)),
                                        ResBlock("{}3".format(name)),
                                        ResBlock("{}4".format(name)),
                                        ResBlock("{}5".format(name)),
                                        ResBlock("{}6".format(name)),
                                        ResBlock("{}7".format(name)),
                                        ResBlock("{}8".format(name)),
                                        ResBlock("{}9".format(name)),
                                        ResBlock("{}10".format(name)),
                                        Decoder(name)
                                        ])
    
    def call(self, x):
        x = self.architecture(x)
        return x

    #@staticmethod
    def loss_fn(self, generation, reconstruction, original):
          G_loss = tf.reduce_mean(tf.square(generation-tf.ones_like(generation)))
          Cycle_loss = tf.reduce_mean(tf.abs(original-reconstruction))
          return G_loss+10*Cycle_loss
    

class Discriminator(Model):

    def __init__(self, args, name):
        super(Discriminator, self).__init__()
    
        # optimizer
        if args:
            self.optimizer = optimizers.Adam(learning_rate=args.lr)

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
        #Reg_loss = tf.reduce_mean(tf.square(mix_original-tf.ones_like(mix_original)))
        #Reg_loss += tf.reduce_mean(tf.square(mix_generation-tf.zeros_like(mix_generation)))
        #return 0.5*D_loss+1.0*0.5*Reg_loss
        return 0.5*D_loss

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
