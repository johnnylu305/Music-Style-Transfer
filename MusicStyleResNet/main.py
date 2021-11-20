import tensorflow as tf
import os
import argparse
import datetime
import numpy as np
from model import Generator, Discriminator
from preprocess import TrainGenerator
from utils import AudioPool, MIDICreator


parser = argparse.ArgumentParser(description='Music Style Transfer')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='../dataset/', help='path of the dataset')
parser.add_argument('--type', dest='type', default='ResNet', help='ResNet')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--decay_step', dest='decay_step', type=int, default=10, help='# of epoch to decay lr')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum for optimizer')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# of video for a batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# epoch')
parser.add_argument('--load_checkpoint', dest='load_checkpoint', default=None, help='load checkpoint')
args = parser.parse_args()

GOPT = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)
DOPT = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)


def train(dataA, dataB, dataABC, genA, genB, disA, disB, disAm, disBm, aud_pool):
    # the length is the smallest one
    for (realA, _), (realB, _), (realABC, _) in zip(dataA, dataB, dataABC): 
        # two tape
        # since we would like to update generator and discriminator respectively
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as dis_tape: 
            # generator
            fakeA = genA(realB, training=True)
            fakeB = genB(realA, training=True)

            recA = genA(fakeB, training=True)
            recB = genB(fakeA, training=True)
            
            # get previous fake data
            his_fakeA, his_fakeB = aud_pool(fakeA, fakeB)
            # get random noise
            noise = np.abs(np.random.normal(size=realA.shape))
            
            # discriminator for A and B
            dis_realA = disA(realA+noise, training=True)
            dis_realB = disB(realB+noise, training=True)

            dis_fakeA = disA(fakeA+noise, training=True)
            dis_fakeB = disB(fakeB+noise, training=True)
            
            dis_his_fakeA = disA(his_fakeA+noise, training=True)
            dis_his_fakeB = disB(his_fakeB+noise, training=True)

            # discriminator for A, B, C
            dis_realAm = disAm(realABC+noise, training=True)
            dis_realBm = disBm(realABC+noise, training=True)
            
            dis_his_fakeAm = disA(his_fakeA+noise, training=True)
            dis_his_fakeBm = disB(his_fakeB+noise, training=True)
            
            # loss
            gen_loss = genA.loss_fn(dis_fakeA, recB, realB)+genB.loss_fn(dis_fakeB, recA, realA)

            dis_loss = disA.loss_fn(dis_realA, dis_his_fakeA)+disB.loss_fn(dis_realB, dis_his_fakeB)
            dis_loss += disAm.loss_fn(dis_realAm, dis_his_fakeAm)+disBm.loss_fn(dis_realBm, dis_his_fakeB) 
            print("Gen:{}, Dis:{}".format(gen_loss.numpy(), dis_loss.numpy()))
        # gradient
        gen_var = genA.trainable_variables+genB.trainable_variables
        dis_var = disA.trainable_variables+disB.trainable_variables
        dis_var += disAm.trainable_variables+disBm.trainable_variables
        
        gradients = gen_tape.gradient(target=gen_loss,
                                  sources=gen_var)
        GOPT.apply_gradients(zip(gradients, gen_var))

        gradients = dis_tape.gradient(target=dis_loss,
                                  sources=dis_var)
        DOPT.apply_gradients(zip(gradients, dis_var))


def main():
    init_epoch = 0
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    # reuse the directory if loading checkpoint
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        timestamp = args.load_checkpoint.split(os.sep)[-2]
    
    save_dir = "./checkpoints/"
    log_dir = "./logs/"
    checkpoint_path = os.path.join(save_dir, args.type, timestamp)
    logs_path = os.path.join(log_dir, args.type, timestamp)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path) 

    # load checkpoints
    if args.load_checkpoint:
        assert os.path.split(args.load_checkpoint)[0]==checkpoint_path
        model.load_weights(args.load_checkpoint)
        init_epoch = int(os.path.split(args.load_checkpoint)[-1].split("-")[0])

    # create data loader
    dataA = TrainGenerator(path=os.path.join(args.dataset_dir,
                                             'preprocess',
                                             'JC_J',
                                             'train'), batch=args.batch_size, shuffle=True)
    dataB = TrainGenerator(os.path.join(args.dataset_dir, 
                                        'preprocess', 
                                        'JC_C', 
                                        'train'), batch=args.batch_size, shuffle=True)
    dataABC = TrainGenerator(os.path.join(args.dataset_dir, 
                                          'preprocess', 
                                          'JCP_mixed'), batch=args.batch_size, shuffle=True)

    

    genA = Generator(args, "GeneratorA")
    genB = Generator(args, "GeneratorB")
    

    disA = Discriminator(args, "DiscriminatorA")
    disB = Discriminator(args, "DiscriminatorB")
    disAm = Discriminator(args, "DiscriminatorAm")
    disBm = Discriminator(args, "DiscriminatorBm")

    aud_pool = AudioPool()    
    for i in range(args.epoch):
         train(dataA, dataB, dataABC, genA, genB, disA, disB, disAm, disBm, aud_pool)

    
    # test
    # midicreator = MIDICreator
    # test = "../dataset/preprocess/CP_C/train/classic_piano_test_1.npy"
    # music = tf.expand_dims(np.load(test).astype(np.float32), axis=0)

    # output = genA(music)

    # midicreator.create_midi_from_piano_rolls(output, "test_output")

if __name__=="__main__":
    main()
