import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential, Input, layers, initializers, activations, optimizers, losses
from module import Encoder, ResBlock, Decoder, AutoEncoder, LSTMConv2DBlock


class LSTMGenerator(Model):

    def __init__(self, args, name):
        super(LSTMGenerator, self).__init__()
    
        # optimizer
        if args:
            self.optimizer = optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)

        # architexture
        """
        nameS = "{}_StyleEocnder".format(name)
        self.styleEncoder = [Encoder(nameS),
                             ResBlock("{}1".format(nameS)),
                             ResBlock("{}2".format(nameS)),
                             ResBlock("{}3".format(nameS)),
                             ResBlock("{}4".format(nameS)),
                             ResBlock("{}5".format(nameS)),
                             ResBlock("{}6".format(nameS)),
                             ResBlock("{}7".format(nameS)),
                             ResBlock("{}8".format(nameS)),
                             ResBlock("{}9".format(nameS)),
                             ResBlock("{}10".format(nameS)),
                             LSTMBlock("{}1".format(nameS)),
                             #Decoder(name)
                             ]
        self.autoEncoder = AutoEncoder("AutoEncoder")
        """
        # architexture
        nameA = "{}_AudioEocnder".format(name)
        self.audioEncoder = [Encoder(nameA),
                             #ResBlock("{}1".format(nameA)),
                             #ResBlock("{}2".format(nameA)),
                             #ResBlock("{}3".format(nameA)),
                             #ResBlock("{}4".format(nameA)),
                             #ResBlock("{}5".format(nameA)),
                             #ResBlock("{}6".format(nameA)),
                             #ResBlock("{}7".format(nameA)),
                             #ResBlock("{}8".format(nameA)),
                             #ResBlock("{}9".format(nameA)),
                             #ResBlock("{}10".format(nameA)),
                             LSTMConv2DBlock("{}1".format(nameA)),
                             Decoder(name)
                             ]
        self.counter = 0
    def call(self, audio, style):
        batch = audio.shape[0]
        style = tf.reshape(style, (-1, 4, 16, 84, 1))
        audio = tf.reshape(audio, (-1, 4, 16, 84, 1))
        """
        for layer in self.styleEncoder:
            if isinstance(layer, LSTMBlock):
                style, encoder_state = layer(style)
            else:
                style = layer(style) 
        encoder_state[0], encoder_state[1] = self.autoEncoder(encoder_state[0]), self.autoEncoder(encoder_state[1])
        """
        
        for layer in self.audioEncoder:
            if isinstance(layer, LSTMConv2DBlock):
                #print(encoder_state[0].shape, encoder_state[1].shape)
                audio, _ = layer(audio)#, initial_state=encoder_state)
            else:
                audio = layer(audio)
        return audio
    
    #@staticmethod
    def loss_fn(self, generation, reconstruction, original):
          G_loss = tf.reduce_mean(tf.square(generation-tf.ones_like(generation)))
          Cycle_loss = tf.reduce_mean(tf.abs(original-reconstruction))
          if self.counter%200==0:
              print(Cycle_loss.numpy())
          self.counter += 1
          return 200*G_loss+Cycle_loss #10*Cycle_loss
    

class Discriminator(Model):

    def __init__(self, args, name):
        super(Discriminator, self).__init__()
    
        # optimizer
        if args:
            self.optimizer = optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)

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
