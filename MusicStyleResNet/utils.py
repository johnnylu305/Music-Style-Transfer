import numpy as np

# ImagePool/AudioPool has been utilized for many GAN-based methods
# It will record the generated images (fake images) for discriminator
# Hence, the discriminator will not forget it
class AudioPool:
    def __init__(self, size=50):
        self.pool_size = size
        self.cur_size = 0
        self.fake_audio = []

    def __call__(self, audioA, audioB):
        if self.cur_size<self.pool_size:
            self.fake_audio.append([audioA, audioB])
            self.cur_size += 1
            return audioA, audioB
        # whether use historical data or not
        if np.random.rand()>0.5:
            idx = np.random.randint(self.pool_size)
            his_audioA, self.fake_audio[idx][0] = self.fake_audio[idx][0], audioA
            idx = np.random.randint(self.pool_size)
            his_audioB, self.fake_audio[idx][1] = self.fake_audio[idx][1], audioB
            return his_audioA, his_audioB
        else:
            return audioA, audioB
