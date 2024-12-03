import os
from mido import MidiFile
import json
import tqdm

FILE_NAME = "seq.json"
# PATH_TO_MIDI = "./dataset/nesmdb/nesmdb_midi"
PATH_TO_MIDI = "./dataset/lmd/"
S_len = 2048
TOKEN_BEGIN = "<BOS>"
TOKEN_END = "<EOS>"
TOKEN_PAD = "<PAD>"


def find_all_mid_files(directory):
    mid_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".mid"):  # 检查是否为 .mid 文件
                mid_files.append(os.path.join(root, file))
    return mid_files


midi_files = find_all_mid_files(PATH_TO_MIDI)


def midi_to_sequence(midi_path):
    '''
    将midi专为token序列，重中之重的一步。
    '''
    
    # def map_pitchwheel(current:int, lowest: int=-8192, highest: int=8192, max_uint: int=127):
    #     import math
    #     range = highest - lowest
    #     position = (current - lowest) / range
    #     return math.floor(position * max_uint)
    
    midi = MidiFile(midi_path)
    sequence = []
    for track in midi.tracks:
        current_time = 0
        for msg in track:
            print(msg)
            if msg.is_meta:
                continue
            if msg.type == "note_on" and msg.velocity > 0:
                sequence.append(f"Note_On_{msg.note}")
                sequence.append(f"VELOCITY_{msg.velocity}")
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                sequence.append(f"Note_Off_{msg.note}")
            elif msg.type == "control_change":
                sequence.append(f"Control_{msg.control}_{msg.value}")
            elif msg.type == "program_change":
                sequence.append(f"Program_{msg.program}")
            elif msg.type == "time_signature":
                sequence.append(f"Time_Signature_{msg.numerator}/{msg.denominator}")
            elif msg.type == "set_tempo":
                sequence.append(f"Tempo_{msg.tempo}")
            elif msg.type == "sysex":
                # 跳过sysex
                pass
            elif msg.type == "pitchwheel":
                # print(msg)
                sequence.append(f"Pitchwheel_{msg.pitch}")
            else:
                raise Exception(f"Unknown message type: {msg.type}")
            
    return sequence


def seq_of(file):
    """
    获取没有经过对齐的，纯字符格式的序列
    """
    return get_all_midi()[file]

def get_all_midi():
    global in_memory_seq
    if len(in_memory_seq) == 0:
        in_memory_seq = load_seq()
    return get_all_midi()

def all_seq_size():
    return len(get_all_midi())

def load_seq():
    with open(FILE_NAME, "r") as f:
        return json.load(f)


def build_seq():
    data = dict()
    midi_files = find_all_mid_files(PATH_TO_MIDI)
    for _file in tqdm.tqdm(midi_files, desc="Building sequences"):
        try:
            data[_file] = {
            "seq": midi_to_sequence(_file),
        }
        except Exception as e:
            print(f"skipping {_file}, cause ",e)
        
    with open(FILE_NAME, "w") as f:
        json.dump(data, f)


def pad_seq(raw, s_len: int = S_len):
    """
    将序列进行对齐
    """
    copy = [note for note in raw]
    copy.insert(0, TOKEN_BEGIN)
    copy = copy[: s_len - 1]
    if len(copy) < s_len:
        copy.append(TOKEN_END)
    while len(copy) < s_len:
        copy.append(TOKEN_PAD)
    return copy
