from dataset.tokenization2 import tokenize2
from seq import tokenize_mid_file
import math

seq = tokenize2("lmd/lmd_full/0/0a0ce238fb8c672549f77f3b692ebf32.mid")
# print(seq)


# def map_pitchwheel(current:int, lowest: int, highest: int, max_uint: int):
#     import math
#     range = highest - lowest
#     position = (current - lowest) / range
#     return math.floor(position * max_uint)


# print(map_pitchwheel(1,-8192, 8192, 200))
