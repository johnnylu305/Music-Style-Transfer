import tensorflow as tf
import os
import argparse
import datetime
import sys
import numpy as np
from tensorflow.keras import callbacks, optimizers, Input
from model import Classifier

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from preprocess import ClassifierTrainGenerator


parser = argparse.ArgumentParser(description='Classifier as a metric')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='../dataset/', help='path of the dataset')
parser.add_argument('--type', dest='type', default='Classifier', help='Classifier')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--decay_step', dest='decay_step', type=int, default=10, help='# of epoch to decay lr')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum for optimizer')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# of video for a batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# epoch')
parser.add_argument('--load_checkpoint', dest='load_checkpoint', default=None, help='load checkpoint')
args = parser.parse_args()


def scheduler(epoch, lr, decay_step=args.decay_step, total_epoch=args.epoch):
    if epoch<decay_step:
        return lr
    else:
        return lr*(total_epoch-epoch)/(total_epoch-decay_step)
OPT = optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)


def train(model, train_generator, val_generator, checkpoint_path, logs_path, init_epoch=0):
    # keras callbacks tensorboard and saver for training
    callback_list = [
        callbacks.TensorBoard(log_dir=logs_path, 
                              update_freq='batch', 
                              profile_batch=0),
        callbacks.ModelCheckpoint(filepath=checkpoint_path+"/{epoch:02d}-{val_accuracy:.3f}.hdf5",
                                  save_weights_only=True,
                                  monitor='val_accuracy',
                                  mode='max',
                                  save_best_only=True,
                                  period=1),
        callbacks.LearningRateScheduler(scheduler)
        ]

    # training
    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        epochs=args.epoch,
        callbacks=callback_list,
        initial_epoch=init_epoch
    )


def test(model, test_generator):
    # test on dataset with labels
    model.evaluate_generator(
        generator=test_generator,
        verbose=1,
    )


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
   
    # define model
    model = Classifier(args, args.type)
    model(Input(shape=(64, 84, 1)))

    # load checkpoints
    if args.load_checkpoint:
        assert os.path.split(args.load_checkpoint)[0]==checkpoint_path
        model.load_weights(args.load_checkpoint)
        init_epoch = int(os.path.split(args.load_checkpoint)[-1].split("-")[0])

    # create data loader
    pathA = '../dataset/preprocess/JC_J/train/'
    pathB = '../dataset/preprocess/JC_C/train/'
    train_gen = ClassifierTrainGenerator(pathA=pathA, 
                                         pathB=pathB, 
                                         A="jazz", 
                                         B="classic", 
                                         batch=args.batch_size, 
                                         shuffle=True,
                                         noise=False)

    pathA = '../dataset/preprocess/JC_J/test/'
    pathB = '../dataset/preprocess/JC_C/test/'
    val_gen = ClassifierTrainGenerator(pathA=pathA, 
                                       pathB=pathB, 
                                       A="jazz", 
                                       B="classic", 
                                       batch=args.batch_size, 
                                       shuffle=True,
                                       noise=True)

    # compile model graph
    model.compile(
        optimizer=OPT,
        loss=model.loss_fn,
        metrics=["accuracy"])

    if args.phase=='train':
        train(model, train_gen, val_gen, checkpoint_path, logs_path, init_epoch)
    else:
        test(model, val_gen)


if __name__=="__main__":
    main()
