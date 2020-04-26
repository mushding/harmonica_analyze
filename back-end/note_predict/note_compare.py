import os
import sys
import note_predict.note_detector.note_detector as note_detector
from note_predict.user_input_sound import UserInputSound
from note_predict.mxl_reader.converter import Converter

class Note_compare:
    def __init__(self):
        self.BPM = 0

    def _getBPM(self):
        return self.BPM

    def bpm_to_sample(self, bpm):
        return int(44100 * (60 / (bpm / 4)))
    
    def generate_correct_timeline(self, mxlname):
        # correct mxl time line
        converter = Converter()
        correct_timeline, self.BPM = converter.reader(os.path.join("./static/HarmonicaData/mxlfile", mxlname))
        return correct_timeline
    
    def generate_user_timeline(self, mxlname, wavname):
        # real user raw input (per 0.1 sec)
        # wav to note array
        # return [[], [], ...] & [, , ...]
        converter = Converter()
        correct_timeline, self.BPM = converter.reader(os.path.join("./static/HarmonicaData/mxlfile", mxlname))
        per_measure_arr, complete_arr = note_detector.detector(os.path.join("./static/HarmonicaData/wav", wavname), self.bpm_to_sample(self.BPM))
        # real user input time line
        user = UserInputSound()
        user_timeline = user.findwrongnote(complete_arr)
        return user_timeline