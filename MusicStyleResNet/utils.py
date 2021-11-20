import numpy as np
import pretty_midi
from tensorflow.keras import optimizers

# ImagePool/AudioPool has been utilized for many GAN-based methods
# It will record the generated images/audio (fake images/audio) for discriminator
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


class LRSchedule(optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, lr, decay_epoch, total_epoch, batch):
        self.lr = lr
        self.decay_step = decay_epoch*batch
        self.total_step = total_epoch*batch
        self.batch = batch

    def __call__(self, step):
        if step<self.decay_step:
            return self.lr
        else:
            return self.lr*(self.total_step-step)/(self.total_step-self.decay_step)


class MIDICreator:
    def __init__(self, tempo=120., beat_resolution=16, pitches=128, velocity=100):
        self.tempo = tempo # bpm
        self.beat_resolution = beat_resolution # time steps per measure
        self.total_num_pitches = pitches
        self.velocity = velocity

        # calculate time step
        seconds_per_beat = 60/tempo
        time_steps_per_beat = beat_resolution/4 # assume 4 beats per measure
        self.time_step = seconds_per_beat/time_steps_per_beat
        


    def create_midi_from_piano_rolls(self, piano_rolls, file_name):
        """
        Creates a midi file with the given filename from a list of piano rolls

        :piano_rolls: Array of piano rolls with shape [N, 64, 84, 1]
        :file_name: File name of exported midi
        """
        
        time_per_roll = np.shape(piano_rolls)[1]*self.time_step

        # create midi
        midi = pretty_midi.PrettyMIDI(initial_tempo = self.tempo)

        for i in range(len(piano_rolls)):
            start_time = i*time_per_roll
            midi = self.write_piano_roll_to_midi(piano_rolls[i], midi, start_time, self.time_step)

        midi.write(file_name+".mid")
        

    def write_piano_roll_to_midi(self, piano_roll, midi, start_time, time_step):

        # create instrument
        piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
        instrument = pretty_midi.Instrument(program=piano_program)

        # figure out how many pitches were cut off
        piano_roll = np.squeeze(piano_roll)    
        pitches_in_piano_roll = np.shape(piano_roll)[1]
        padding = int((self.total_num_pitches - pitches_in_piano_roll)/2)

        # loop through time
        note_on = np.zeros(self.total_num_pitches, dtype=bool) # keeps track of whether a note is on or off right now. (pitch: status)
        note_starts = np.zeros(self.total_num_pitches) # keeps track of most recent note-on time

        for t in range(len(piano_roll)):
            notes_now = piano_roll[t]

            for p in range(len(notes_now)):
                pitch = p + padding
                pitch_on = notes_now[p]

                # if pitch has changed status
                if(pitch_on != note_on[pitch]):
                    if(pitch_on): # starting a new note
                        note_starts[pitch] = t
                    else: # ending a note
                        note_start = (note_starts[pitch])*time_step + start_time
                        note_end = t*time_step + start_time
                        note = pretty_midi.Note(velocity=self.velocity, pitch=pitch, start=note_start, end=note_end)
                        instrument.notes.append(note)
                    note_on[pitch] = pitch_on


        midi.instruments.append(instrument)

        return midi

def main():
    test_1 = "../dataset/preprocess/CP_C/train/classic_piano_train_1.npy"
    test_2 = "../dataset/preprocess/CP_C/train/classic_piano_train_2.npy"

    part1 = np.load(test_1).astype(np.float32)
    part2 = np.load(test_2).astype(np.float32)

    midi_creator = MIDICreator()

    midi_creator.create_midi_from_piano_rolls([part1, part2], "classic_piano_train_1")

if __name__=="__main__":
    main()
