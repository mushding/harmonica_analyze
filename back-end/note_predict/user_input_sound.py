import numpy as np
class UserInputSound:
    def findwrongnote(self, output):
        NOTE_NAMES = 'C C# D D# E F F# G G# A A# B'.split()
        outputarray = list()

        pre_index = 0
        for index in range(1, len(output) + 1):
            time_dict = dict()
            if index == len(output):
                time_dict['start'] = round(pre_index, 1)
                time_dict['end'] = round(0.1 * (index - 1), 1)
                time_dict['type'] = NOTE_NAMES[output[index - 1] % 12] + str(int(output[index - 1]/12 - 1))
                time_dict['resize'] = False
                time_dict['drag'] = False
                outputarray.append(time_dict)
            elif output[index - 1] != output[index]:
                time_dict['start'] = round(pre_index, 1)
                time_dict['end'] = round(0.1 * (index - 1), 1)
                time_dict['type'] = NOTE_NAMES[output[index - 1] % 12] + str(int(output[index - 1]/12 - 1))
                time_dict['resize'] = False
                time_dict['drag'] = False
                pre_index = 0.1 * (index - 1)
                outputarray.append(time_dict)
                tmp = output[index]
        return outputarray




                









