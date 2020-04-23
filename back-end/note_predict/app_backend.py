from open_track import Open_track
from load_model import CNN ,predict 
import sys
sys.path.append('./bpm_detecotr')
from bpm_detection import Bpm_detector

FILENAME = "little_star.wav"

# open track
open = Open_track()
tensor_track, sample_rate = open.readsound(FILENAME)

# detect bpm
detector = Bpm_detector()
bpm = detector.main(FILENAME, 3)
print(bpm)
exit()

# predict with model
model = open.loadmodel()
output = open.putincnn(tensor_track, model)
print(output)
# Dout = open.findwrongnote(output)