#! /usr/bin/env python
######################################################################
# tuner.py - a minimal command-line guitar/ukulele tuner in Python.
# Requires numpy and pyaudio.
######################################################################
# Author:  Matt Zucker
# Date:    July 2016
# License: Creative Commons Attribution-ShareAlike 3.0
#          https://creativecommons.org/licenses/by-sa/3.0/us/
######################################################################

import os
import numpy as np
from scipy.io import wavfile
import pyaudio

######################################################################
# Feel free to play with these numbers. Might want to change NOTE_MIN
# and NOTE_MAX especially for guitar/bass. Probably want to keep
# FRAME_SIZE and FRAMES_PER_FFT to be powers of two.

NOTE_MIN = 48       # C3
NOTE_MAX = 83       # B5
FSAMP = 44100       # Sampling frequency in Hz
FRAME_SIZE = 4410   # How many samples per frame?
FRAMES_PER_FFT = 2  # FFT takes average across how many frames?

######################################################################
# Derived quantities from constants above. Note that as
# SAMPLES_PER_FFT goes up, the frequency step size decreases (so
# resolution increases); however, it will incur more delay to process
# new sounds.

SAMPLES_PER_FFT = FRAME_SIZE*FRAMES_PER_FFT
FREQ_STEP = float(FSAMP)/SAMPLES_PER_FFT

######################################################################
# For printing out notes

NOTE_NAMES = 'C C# D D# E F F# G G# A A# B'.split()

######################################################################
# These three functions are based upon this very useful webpage:
# https://newt.phys.unsw.edu.au/jw/notes.html

def freq_to_number(f): return 69 + 12*np.log2(f/440.0)
def number_to_freq(n): return 440 * 2.0**((n-69)/12.0)
def note_name(n): return NOTE_NAMES[n % 12] + str(n/12 - 1)

######################################################################
# Ok, ready to go now.

# Get min/max index within FFT of notes we care about.
# See docs for numpy.rfftfreq()
def note_to_fftbin(n): return number_to_freq(n)/FREQ_STEP

def detector(filename, measure_sample):
    imin = max(0, int(np.floor(note_to_fftbin(NOTE_MIN-1))))
    imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX+1))))

    # Allocate space to run an FFT. 
    buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
    num_frames = 0
    note_array = []
    tmp = []

    # Create Hanning window function
    window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, SAMPLES_PER_FFT, False)))

    # Open Wavfile
    sample_rate, waveform = wavfile.read(filename)
    # waveform = waveform[..., 0]
    # waveform = waveform[0, :]

    # Print initial text
    # print('sampling at', FSAMP, 'Hz with max resolution of', FREQ_STEP, 'Hz')
    
    complete_note_array = []
    # ignoreStartFlag = True
    for measure in range(0, len(waveform), measure_sample):
        tmp = []
        waveform_part = waveform[measure: measure + measure_sample]
        if len(waveform_part) < measure_sample:
            break
        for index in range(measure, measure + measure_sample, FRAME_SIZE):
            # skip when there is no sound
            # if ignoreStartFlag:
            #     if max(waveform[index: index + FRAME_SIZE]) < 1000:
            #         ignoreStartFlag = False
            #         continue
            # Shift the buffer down and new data in
            try:
                buf[:-FRAME_SIZE] = buf[FRAME_SIZE:]
                buf[-FRAME_SIZE:] = waveform[index: index + FRAME_SIZE]
            except:
                # print("finish")
                break

            # Run the FFT on the windowed buffer
            fft = np.fft.rfft(buf * window)

            # Get frequency of maximum response in range
            freq = (np.abs(fft[imin:imax]).argmax() + imin) * FREQ_STEP

            # Get note number and nearest note
            n = freq_to_number(freq)
            n0 = int(round(n))
            tmp.append(n0)
            complete_note_array.append(n0)
        note_array.append(tmp)
    return note_array, complete_note_array
