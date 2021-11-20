from tensorflow.keras.utils import Sequence
import numpy as np   
import glob
import os

class ClassifierTrainGenerator(Sequence):

    def __init__(self, pathA, pathB, A, B, batch, shuffle=False, noise=False):
        self.batch = batch
        self.shuffle = shuffle
        self.paths, self.idxs = self.load_ids(pathA, pathB)
        self.nameToCat = {A:0, B:1}
        self.noise = noise

    def __len__(self):
        # number of iteration for one epoch
        # drop the last batch if non-divisible
        self.on_epoch_begin()
        return np.floor(len(self.paths)/self.batch).astype(int)

    def __getitem__(self, idx):
        # get indexs for this batch
        idxs = self.idxs[idx*self.batch:(idx+1)*self.batch]
        self.xs, self.ys = self.__data_generation(idxs)
        if self.noise == True:
            self.xs += np.random.normal(size=self.xs.shape)
        return self.xs, self.ys

    def load_ids(self, pathA, pathB):
        # get paths of files
        paths = np.array(glob.glob(os.path.join(pathA, '*.npy'))+glob.glob(os.path.join(pathB, '*.npy')))
        idxs = np.arange(len(paths))
        return paths, idxs

    def on_epoch_begin(self):
        if self.shuffle == True:
            np.random.shuffle(self.idxs)

    def __data_generation(self, idxs):
        # (batch, timestep*consecutive block, notes, channel)
        xs = np.zeros((self.batch, 64, 84, 1))
        ys = np.zeros(self.batch)
        for i, path in enumerate(np.array(self.paths)[idxs]):
            music = np.load(path).astype(np.float32)
            xs[i,:,:,:] = music
            ys[i] = self.nameToCat[os.path.split(path)[-1].split("_")[0]]
        return xs, ys


if __name__=="__main__":
    pathA = '../dataset/preprocess/JC_J/train/'
    pathB = '../dataset/preprocess/JC_C/train/'
    JC = TrainGenerator(pathA=pathA, pathB=pathB, A="jazz", B="classic", batch=5, shuffle=True)
    for xs, ys in JC:
        print(xs.shape)
        print(ys)

