#!/usr/bin/env python

from fractions import Fraction
from note_predict.mxl_reader.reader import Measure

# define the bass part is at the 3rd line
BASS_PART = 5
CHORD_PART = 4
PAGE_PER_COLUMN = 2
IS_FLAT_TO_SHARP = True

# note_array for classification
note_array_measure = []
note_array = []

accidentList = [[['']*8 for i in range(8)]]

crossMeasureTie = False

STEP_TO_NUMBER = {
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'A': 6,
    'B': 7
}

def stepToNumber(step):
    return str(STEP_TO_NUMBER[step])

def stepToNumberFlat(step):
    if STEP_TO_NUMBER[step] == 1:
        return "7"
    return str(STEP_TO_NUMBER[step] - 1)

def generateOctaveMark(octave, isBass=False):
    return " " + str(round(octave/12 - 1) + 6)

def generateTimeSuffix(duration, divisions):
    note_length = Fraction(duration, divisions)
    if duration < divisions: # less than quarter notes: add / and continue
        return "/" + generateTimeSuffix(duration*2, divisions)
    elif duration == divisions: # quarter notes
        return ""
    elif duration * 2 == divisions * 3: # syncopated notes
        return "."
    else: # sustained more than 1.5 quarter notes: add - and continue
        return "-" + generateTimeSuffix(duration - divisions, divisions)

def getNoteDisplayedDuration(note):
    if note.isTuplet():
        return note.getDisplayedDuration()
    else:
        return note.getDuration()

NOTE_DEGREE_TABLE = {
    'C': 0, 'B#': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4,
    'F': 5, 'E#': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11
}


ACCIDENTAL_TABLE = {
    'C':  ('#', []),
    'G':  ('#', ['F']),
    'D':  ('#', ['F', 'C']),
    'A':  ('#', ['F', 'C', 'G']),
    'E':  ('#', ['F', 'C', 'G', 'D']),
    'B':  ('#', ['F', 'C', 'G', 'D', 'A']),
    'F#': ('#', ['F', 'C', 'G', 'D', 'A', 'E']),
    'F':  ('b', ['B']),
    'Bb': ('b', ['B', 'E']),
    'Eb': ('b', ['B', 'E', 'A']),
    'Ab': ('b', ['B', 'E', 'A', 'D']),
    'Db': ('b', ['B', 'E', 'A', 'D', 'G']),
    'Gb': ('b', ['B', 'E', 'A', 'D', 'G', 'C']),
}

DEGREE_NOTE_TABLE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def getTransposedPitch(note_name, octave, offset):
    degree = NOTE_DEGREE_TABLE[note_name]
    transposed_degree = degree + offset
    transposed_octave = octave + transposed_degree // 12
    transposed_degree %= 12
    return (DEGREE_NOTE_TABLE[transposed_degree], transposed_octave)

def getTransposeOffsetToC(key):
    degree = NOTE_DEGREE_TABLE[key]
    if degree <= 6:
        return -degree
    else:
        return 12 - degree

def generateBasicNote(note, isBass=False):
    global accidentList

    (duration, divisions) = getNoteDisplayedDuration(note)
    time_suffix = " " + generateTimeSuffix(duration, divisions)
    
    # Turn flat into all sharp
    if note.isRest():
        return "0" + time_suffix
    else:
        pitch = note.getPitch()
        (note_name, octave) = note.getPitch()

        keysig = note.getAttributes().getKeySignature()
        step = note_name[0:1] # C, D, E, F, G, A, B
        accidental = note_name[1:2] # sharp (#) and flat (b)
        force_accidental = note_name[2:3] # additonal sharp and flat and natural

        key_accidental_char, key_accidental_list = ACCIDENTAL_TABLE[keysig]

        if step in key_accidental_list:
            if force_accidental != "":
                accidentList[STEP_TO_NUMBER[step]][octave] = accidental
            elif accidentList[STEP_TO_NUMBER[step]][octave] == '=':
                accidental = '='
            accidentList[STEP_TO_NUMBER[step]][octave] = accidental

        elif accidental == "":
            preNote = STEP_TO_NUMBER[step] + 1
            if preNote == 8:
                preNote = 1
            if accidentList[preNote][octave] == 'b' and IS_FLAT_TO_SHARP and accidentList[STEP_TO_NUMBER[step]][octave] == '':
                accidental = '='
                accidentList[STEP_TO_NUMBER[step]][octave] = accidental
            else:
                accidental = accidentList[STEP_TO_NUMBER[step]][octave]
        else:
            accidentList[STEP_TO_NUMBER[step]][octave] = accidental

        if accidental == 'b' and IS_FLAT_TO_SHARP:
            if step == 'C':
                return step + generateOctaveMark(octave - 1, isBass) + time_suffix
            elif step == 'F':
                return step + generateOctaveMark(octave, isBass) + time_suffix
            accidental = '#'
            return step + accidental + generateOctaveMark(octave, isBass) + time_suffix
        else:
            return step + accidental + generateOctaveMark(octave, isBass) + time_suffix

def generateNote(note, isBass=False):
    global crossMeasureTie
    result = generateBasicNote(note, isBass)
    note_array_measure.append(result)
    return result

def generateMeasure(measure, isBass=False):
    global accidentList
    global crossMeasureTie

    if not crossMeasureTie:
        accidentList = [['']*8 for i in range(8)]

    pieces = [generateNote(note, isBass) for note in measure]
    return ' '.join(pieces)

def generateRightBarline(measure):
    if measure.getRightBarlineType() == Measure.BARLINE_REPEAT:
        return ":|"
    elif measure.getRightBarlineType() == Measure.BARLINE_DOUBLE:
        return "||/"
    elif measure.getRightBarlineType() == Measure.BARLINE_FINAL:
        return "||"
    else:
        return "|"

def generateMeasures(measureList, isBass=False):
    global note_array_measure
    pieces = []
    for i, measure in enumerate(measureList):
        note_array_measure = []
        pieces.append(" ")                              
        pieces.append(generateMeasure(measure, isBass))             # see content
        pieces.append(" ")
        note_array.append(note_array_measure)
    return ''.join(pieces)

def getSecPerMeasure(reader):
    return (60 / (int(reader.getBPM()) / 4)) / 4

def generateBody(reader, max_measures_per_line=4):

    global accidentList
    parts = reader.getPartIdList()

    part_measures = dict()
    for part in parts:
        part_measures[part] = list(reader.iterMeasures(part))

    lines = []
    column_now = 0

    measure_count = max(len(measures) for measures in part_measures.values())
    for i in range(0, measure_count, max_measures_per_line):
        begin = i
        end = min(i + max_measures_per_line, measure_count)
        for part_index, part in enumerate(parts):
            accidentList = [['']*8 for i in range(8)]
            line = ""
            if part_index + 1 == BASS_PART:
                line += generateMeasures(part_measures[part][begin:end], isBass = True)   # enter bass part
            else:   
                line += generateMeasures(part_measures[part][begin:end])   # other part
            lines.append(line)
        lines.append('') # empty line
        column_now = column_now + 1

    return '\n'.join(lines)

def generateDuration(timeSigArr):
    duration = 1
    for timeSig in timeSigArr:
        if timeSig == '-':
            duration += 1
        elif timeSig == '/':
            duration *= 0.5
        elif timeSig == '.':
            duration = duration + duration * 0.5
    return duration

def generateDictionary(reader):
    sec_per_note = getSecPerMeasure(reader)
    mxl_time = list()
    note_sec = 0
    for measure in note_array:
        for note in measure:
            tmp_dir = dict()
            tmp_dir['start'] = note_sec
            index = note.split(" ")
            note_sec = note_sec + generateDuration(index[2]) * sec_per_note
            tmp_dir['end'] = note_sec
            tmp_dir['type'] = index[0] + index[1]
            tmp_dir['resize'] = False
            tmp_dir['drag'] = False
            mxl_time.append(tmp_dir)
    return mxl_time

class WriterError(Exception):
    pass

class Jianpu99Writer:

    def generate(self, reader):
        generateBody(reader)
        mxl_time = generateDictionary(reader)
        return mxl_time
