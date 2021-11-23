import tensorflow as tf
import os
import argparse
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from model import LSTMGenerator, Discriminator, Classifier
from preprocess import TrainGenerator, TestGenerator, ClassifierGenerator
from utils import AudioPool, MIDICreator, LRSchedule, get_saver, get_writer 


parser = argparse.ArgumentParser(description='Music Style Transfer')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='../dataset/', help='path of the dataset')
parser.add_argument('--type', dest='type', default='LSTM', help='LSTM/Transformer')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--decay_step', dest='decay_step', type=int, default=10, help='# of epoch to decay lr')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum for optimizer')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# of video for a batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# epoch')
parser.add_argument('--load_checkpoint', dest='load_checkpoint', default=None, help='load checkpoint')
parser.add_argument('--load_classifier', dest='load_classifier', default=None, help='load checkpoint for classifier')
parser.add_argument('--generate_midi', dest='generate_midi', default=0, help='number of epochs between generating midi files (0 means none are generated)')
args = parser.parse_args()

MAX_S = 0
ITER = 11216//args.batch_size
GOPT = tf.keras.optimizers.Adam(learning_rate=LRSchedule(args.lr, args.decay_step, args.epoch, ITER), 
                                beta_1=args.beta1)
DOPT = tf.keras.optimizers.Adam(learning_rate=LRSchedule(args.lr, args.decay_step, args.epoch, ITER), 
                                beta_1=args.beta1)


def train(dataA, dataB, dataABC, genA, genB, disA, disB, disAm, disBm, aud_pool, epoch, writer):
    gen_losses = []
    dis_losses = []
    # the length is the smallest one
    for i, ((realA, _), (realB, _), (realABC, _)) in enumerate(zip(dataA, dataB, dataABC)): 
        # two tape
        # since we would like to update generator and discriminator respectively
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as dis_tape: 
            # generator
            fakeA = genA(realB, realA, training=True)
            fakeB = genB(realA, realB, training=True)

            recA = genA(fakeB, realA, training=True)
            recB = genB(fakeA, realB, training=True)
            
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
            gen_losses.append(gen_loss)
            dis_losses.append(dis_loss)
            if i%400==0:
                print("Gen: {:.3f}, Dis: {:.3f}".format(gen_loss.numpy(), dis_loss.numpy()))
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

    with writer.as_default():
        tf.summary.scalar('Generator Loss', np.mean(gen_losses), step=epoch)
        tf.summary.scalar('Discriminator Loss', np.mean(dis_losses), step=epoch)


def test(classifier, genA, genB, test_genA, test_genB, epoch, writer, saver, checkpoint_path, midi_path):
    acc_A = tf.keras.metrics.Accuracy()
    acc_AB = tf.keras.metrics.Accuracy()
    acc_ABA = tf.keras.metrics.Accuracy()
    acc_B = tf.keras.metrics.Accuracy()
    acc_BA = tf.keras.metrics.Accuracy()
    acc_BAB = tf.keras.metrics.Accuracy()
    # make sure test_genA and test_genB have the same number of data
    for (xs_A, ys_A), (xs_B, ys_B) in zip(test_genA, test_genB):
        acc_A.update_state(classifier(xs_A)>0.5, ys_A)
        AB = genB(xs_A, xs_B)
        acc_AB.update_state(classifier(AB)>0.5, ys_A)
        ABA = genA(AB, xs_A)
        acc_ABA.update_state(classifier(ABA)>0.5, ys_A)

        acc_B.update_state(classifier(xs_B)>0.5, ys_B)
        BA = genA(xs_B, xs_A)
        acc_BA.update_state(classifier(BA)>0.5, ys_B)
        BAB = genB(BA, xs_B)
        acc_BAB.update_state(classifier(BAB)>0.5, ys_B)
    
    acc_A = acc_A.result().numpy()
    acc_AB = acc_AB.result().numpy()
    acc_ABA = acc_ABA.result().numpy()
    SA = (acc_A+acc_ABA-2*acc_AB)/2
    print("A: {:.3f}, AB: {:.3f}, ABA: {:.3f}, SA: {:.3f}".format(acc_A, acc_AB, acc_ABA, SA))

    acc_B = acc_B.result().numpy()
    acc_BA = acc_BA.result().numpy()
    acc_BAB = acc_BAB.result().numpy()
    SB = (acc_B+acc_BAB-2*acc_BA)/2
    print("B: {:.3f}, BA: {:.3f}, BAB: {:.3f}, SB: {:.3f}".format(acc_B, acc_BA, acc_BAB, SB))
    S = (SA+SB)/2
    print("S: {:.3f}".format(S))
    
    # save weights
    global MAX_S
    if saver and (S>MAX_S or epoch%5==0):
        MAX_S = S
        saver.save(os.path.join(checkpoint_path, '{:03d}-{:.3f}').format(epoch, S))
    
    # tensorbaord
    if writer:
        with writer.as_default():
            tf.summary.scalar('acc_A', acc_A, step=epoch)
            tf.summary.scalar('acc_AB', acc_AB, step=epoch)
            tf.summary.scalar('acc_ABA', acc_ABA, step=epoch)
            tf.summary.scalar('SA', SA, step=epoch)
            tf.summary.scalar('acc_B', acc_B, step=epoch)
            tf.summary.scalar('acc_BA', acc_BA, step=epoch)
            tf.summary.scalar('acc_BAB', acc_BAB, step=epoch)
            tf.summary.scalar('SB', SB, step=epoch)
            tf.summary.scalar('S', S, step=epoch)

    # generate midi files
    # only the first sample from each dataset is tested
    if(int(args.generate_midi) > 0 and epoch%int(args.generate_midi) == 0):
        midicreator = MIDICreator()

        # generate B from A
        A_songs, _ = test_genA[0]
        B_songs, _ = test_genB[0]
        AB_1 = genB(A_songs[0:1], B_songs[0:1])
        ABfilename = "AB" + str(epoch)
        print(ABfilename)
        midicreator.create_midi_from_piano_rolls(AB_1, os.path.join(midi_path, ABfilename))

        # generate A from B
        BA_1 = genA(B_songs[0:1], A_songs[0:1])
        BAfilename = "BA" + str(epoch)
        midicreator.create_midi_from_piano_rolls(BA_1, os.path.join(midi_path, BAfilename))        


def main():
    init_epoch = 0
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

    # reuse the directory if loading checkpoint
    if args.load_checkpoint and os.path.exists(os.path.split(args.load_checkpoint)[0]):
        timestamp = args.load_checkpoint.split(os.sep)[-2]
    

    save_dir = "./checkpoints/"
    log_dir = "./logs/"
    midi_path = "./midi/"

    checkpoint_path = os.path.join(save_dir, args.type, timestamp)
    logs_path = os.path.join(log_dir, args.type, timestamp)
    midi_path = os.path.join(midi_path, args.type, timestamp)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path) 
    if not os.path.exists(midi_path):
        os.makedirs(midi_path) 

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

    
    if args.type=='LSTM':
        genA = LSTMGenerator(args, "LSTMGeneratorA")
        genB = LSTMGenerator(args, "LSTMGeneratorB")
    else:
        genA = LSTMGenerator(args, "GeneratorA")
        genB = LSTMGenerator(args, "GeneratorB")       
  
    disA = Discriminator(args, "DiscriminatorA")
    disB = Discriminator(args, "DiscriminatorB")
    disAm = Discriminator(args, "DiscriminatorAm")
    disBm = Discriminator(args, "DiscriminatorBm")


    # get saver
    saver = get_saver(GOPT, DOPT, genA, genB, disA, disB, disAm, disBm, checkpoint_path)
    # load checkpoints
    if args.load_checkpoint:
        global MAX_S
        assert os.path.samefile(os.path.split(args.load_checkpoint)[0], checkpoint_path)
        if saver.restore(args.load_checkpoint): #.expect_partial():
            # reset epoch
            init_epoch = int(os.path.split(args.load_checkpoint)[-1].split("-")[0])+1
            MAX_S = float(os.path.split(args.load_checkpoint)[-1].split("-")[1])
            print("Load checkpoint succeeded")
        #init_epoch = int(os.path.split(args.load_checkpoint)[-1].split("-")[0])

    # get tensorboard writer
    writer = get_writer(logs_path)


    if args.load_classifier:
        classifier = Classifier(None, "Classifier")
        classifier(Input(shape=(64, 84, 1)))
        classifier.load_weights(args.load_classifier)
        pathA = '../dataset/preprocess/JC_J/test/'
        pathB = '../dataset/preprocess/JC_C/test/'            
        val_genA = TestGenerator(pathA=pathA, 
                                 pathB=None, 
                                 A="jazz", 
                                 B="classic", 
                                 batch=args.batch_size, 
                                 shuffle=False)
        val_genB = TestGenerator(pathA=None, 
                                 pathB=pathB, 
                                 A="jazz", 
                                 B="classic", 
                                 batch=args.batch_size, 
                                 shuffle=False)

        # compile model graph
        classifier.compile(
            loss=classifier.loss_fn,
            metrics=["accuracy"])

    aud_pool = AudioPool()    
    if args.phase=='train':
        for i in range(args.epoch):
            train(dataA, dataB, dataABC, genA, genB, disA, disB, disAm, disBm, aud_pool, i, writer)
            if args.load_classifier:
                test(classifier, genA, genB, val_genA, val_genB, i, writer, saver, checkpoint_path, midi_path)
    else:
        test(classifier, genA, genB, val_genA, val_genB, None, None, None, checkpoint_path, midi_path)


if __name__=="__main__":
    main()
