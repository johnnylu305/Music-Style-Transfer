from tensorflow.keras.utils import Sequence
import numpy as np   
import glob
import os

class TrainGenerator(Sequence):

    def __init__(self, path, batch, shuffle=False):
        self.batch = batch
        self.shuffle = shuffle
        self.paths, self.idxs = self.load_ids(path)
        self.nameToCat = {"classic":0, "pop":1, "jazz":2}

    def __len__(self):
        # number of iteration for one epoch
        # drop the last batch if non-divisible
        self.on_epoch_begin()
        return np.floor(len(self.paths)/self.batch).astype(int)

    def __getitem__(self, idx):
        # get indexs for this batch
        idxs = self.idxs[idx*self.batch:(idx+1)*self.batch]
        self.xs, self.ys = self.__data_generation(idxs)
        return self.xs, self.ys

    def load_ids(self, path):
        # get paths of files
        paths = np.array(glob.glob(os.path.join(path, '*.npy')))
        idxs = np.arange(len(paths))
        return paths, idxs

    def on_epoch_begin(self):
        if self.shuffle == True:
            np.random.shuffle(self.idxs)

    def __data_generation(self, idxs):
        xs = np.zeros((self.batch, 64, 84))
        ys = np.zeros(self.batch)
        for i, path in enumerate(np.array(self.paths)[idxs]):
            music = np.load(path).astype(np.float32)
            xs[i,:,:] = music[:,:,0]
            ys[i] = self.nameToCat[os.path.split(path)[-1].split("_")[0]]
        return xs, ys


if __name__=="__main__":
    CP_C = TrainGenerator(path='./dataset/preprocess/CP_C/train/', batch=5, shuffle=True)
    for xs, ys in CP_C:
        print(xs.shape)
        print(ys.shape)
    CP_P = TrainGenerator(path='./dataset/preprocess/CP_P/train/', batch=5, shuffle=True)
    for xs, ys in CP_C:
        print(xs.shape)
        print(ys.shape)
    JCP_MIX = TrainGenerator(path='./dataset/preprocess/JCP_mixed/', batch=5, shuffle=True)
    for xs, ys in JCP_MIX:
        print(xs.shape)
        print(ys)
