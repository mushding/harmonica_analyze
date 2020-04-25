
#!/usr/bin/env python3

import argparse
import os

from note_predict.mxl_reader.reader import MusicXMLReader, MusicXMLParseError
from note_predict.mxl_reader.writer import Jianpu99Writer, WriterError

class Converter:
    def reader(self, filename):
        reader = MusicXMLReader(filename)
        writer = Jianpu99Writer()

        try:
            return writer.generate(reader), int(reader.getBPM())
        except WriterError as e:
            print("error: %s" % str(e))
