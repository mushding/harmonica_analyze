# Copyright 2012 Free Software Foundation, Inc.
#
# This file is part of The BPM Detector Python
#
# The BPM Detector Python is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# The BPM Detector Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the,
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with The BPM Detector Python; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

import argparse
import array
import math
import time
import wave

import matplotlib.pyplot as plt
import numpy
import pywt
from scipy import signal

class Bpm_detector:

    def read_wav(self, filename):
        # open file, get metadata for audio
        try:
            wf = wave.open(filename, 'rb')
        except IOError as e:
            print(e)
            return

        # typ = choose_type( wf.getsampwidth() ) #TODO: implement choose_type
        nsamps = wf.getnframes()
        assert (nsamps > 0)

        fs = wf.getframerate()
        assert (fs > 0)

        # read the entire file and make into an array
        samps = list(array.array('i', wf.readframes(nsamps)))
        # print 'Read', nsamps,'samples from', filename
        try:
            assert (nsamps == len(samps))
        except AssertionError:
            print(nsamps, "not equal to", len(samps))

        return samps, fs


    # print an error when no data can be found
    def no_audio_data(self):
        print("No audio data for sample, skipping...")
        return None, None


    # simple peak detection
    def peak_detect(self, data):
        max_val = numpy.amax(abs(data))
        peak_ndx = numpy.where(data == max_val)
        if len(peak_ndx[0]) == 0:  # if nothing found then the max must be negative
            peak_ndx = numpy.where(data == -max_val)
        return peak_ndx


    def bpm_detector(self, data, fs):
        cA = []
        cD_minlen = -1
        cD_sum = []
        levels = 4
        max_decimation = 2 ** (levels - 1)
        min_ndx = math.ceil(60. / 220 * (fs / max_decimation))
        max_ndx = math.ceil(60. / 40 * (fs / max_decimation))

        for loop in range(0, levels):
            # 1) DWT
            if loop == 0:
                [cA, cD] = pywt.dwt(data, 'db4')
                cD_minlen = math.floor(len(cD) / max_decimation + 1)
                cD_sum = numpy.zeros(cD_minlen)
            else:
                [cA, cD] = pywt.dwt(cA, 'db4')
            # 2) Filter
            cD = signal.lfilter([0.01], [1 - 0.99], cD)

            # 4) Subtract args.filename out the mean.

            # 5) Decimate for reconstruction later.
            cD = abs(cD[::(2 ** (levels - loop - 1))])
            cD = cD - numpy.mean(cD)
            # 6) Recombine the signal before ACF
            #    essentially, each level I concatenate 
            #    the detail coefs (i.e. the HPF values)
            #    to the beginning of the array
            cD_sum = cD[:cD_minlen] + cD_sum

        if not [b for b in cA if b != 0.0]:
            return self.no_audio_data()
        # adding in the approximate data as well...    
        cA = signal.lfilter([0.01], [1 - 0.99], cA)
        cA = abs(cA)
        cA = cA - numpy.mean(cA)
        cD_sum = cA[0:cD_minlen] + cD_sum

        # ACF
        correl = numpy.correlate(cD_sum, cD_sum, 'full')

        midpoint = math.ceil(len(correl) / 2)
        correl_midpoint_tmp = correl[midpoint:]
        peak_ndx = self.peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])
        if len(peak_ndx) > 1:
            return self.no_audio_data()

        peak_ndx_adjusted = peak_ndx[0] + min_ndx
        bpm = 60. / peak_ndx_adjusted * (fs / max_decimation)
        print(bpm)
        return bpm, correl


    def main(self, filename, window):
        filename = "../../dataset/harmonica_test/" + filename
        samps, fs = self.read_wav(filename)

        correl = []
        n = 0
        nsamps = len(samps)
        window_samps = math.ceil(window * fs)
        samps_ndx = 0  # first sample in a window_ndx
        max_window_ndx = math.ceil(nsamps / window_samps)
        bpms = numpy.zeros(max_window_ndx)

        # iterate through all windows
        for window_ndx in range(max_window_ndx):

            # get a new set of samples
            # print n,":",len(bpms),":",max_window_ndx,":",fs,":",nsamps,":",samps_ndx
            data = samps[samps_ndx:samps_ndx + window_samps]
            # if not ((len(data) % window_samps) == 0):
            #     raise AssertionError(str(len(data)))

            bpm, correl_temp = self.bpm_detector(data, fs)
            if bpm is None:
                continue
            bpms[window_ndx] = bpm
            correl = correl_temp

            # iterate at the end of the loop
            samps_ndx = samps_ndx + window_samps
            n = n + 1  # counter for debug...

        bpm = numpy.median(bpms)
        print('Completed.  Estimated Beats Per Minute:', bpm)
        return bpm