import os
import sys
import note_predict.note_detector.note_detector as note_detector
from note_predict.user_input_sound import UserInputSound
from note_predict.mxl_reader.converter import Converter

class Note_compare:
    def bpm_to_sample(self, bpm):
        return int(44100 * (60 / (bpm / 4)))
    
    def generate_correct_user_timeline(self):

        # correct mxl time line
        converter = Converter()
        mxlname = "tests.mxl"
        correct_timeline, bpm = converter.reader(os.path.join("./note_predict/mxl_reader/mxl_file_data", mxlname))
        print(correct_timeline)

        # real user raw input (per 0.1 sec)
        # wav to note array
        # return [[], [], ...] & [, , ...]
        wavname = "tests.wav"
        per_measure_arr, complete_arr = note_detector.detector(os.path.join("./static/HarmonicaData/wav", wavname), self.bpm_to_sample(bpm))
        print(complete_arr)

        # real user input time line
        user = UserInputSound()
        user_timeline = user.findwrongnote(complete_arr)
        print(user_timeline)