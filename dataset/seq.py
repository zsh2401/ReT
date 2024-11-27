import os
from mido import MidiFile
import json
import tqdm
FILE_NAME = "seq.json"

def find_all_mid_files(directory):
    mid_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mid'):  # 检查是否为 .mid 文件
                mid_files.append(os.path.join(root, file))
    return mid_files


def midi_to_sequence(midi_path):
    midi = MidiFile(midi_path)
    sequence = []
    for track in midi.tracks:
        for msg in track:
            if msg.is_meta:
                continue
            if msg.type == 'note_on' and msg.velocity > 0:
                sequence.append(f"Note_On_{msg.note}")
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                sequence.append(f"Note_Off_{msg.note}")
            elif msg.type == 'control_change':
                sequence.append(f"Control_{msg.control}_{msg.value}")
            elif msg.type == 'program_change':
                sequence.append(f"Program_{msg.program}")
            elif msg.type == 'time_signature':
                sequence.append(f"Time_Signature_{msg.numerator}/{msg.denominator}")
            elif msg.type == 'set_tempo':
                sequence.append(f"Tempo_{msg.tempo}")
            else:
                raise Exception(f"Unknown message type: {msg.type}")
    return sequence


CX = dict()

def seq_of(file):
    global CX
    if len(CX) == 0:
        CX = load_seq()
    return CX[file]

def load_seq():
    with open(FILE_NAME,"r") as f:
        return json.load(f)
    
def build_seq():
    data = dict()
    for _file in tqdm.tqdm(midi_files, desc="Building sequences"):
            data[_file] = midi_to_sequence(_file)
    
    with open(FILE_NAME,"w") as f:
        json.dump(data,f)
        
midi_files = find_all_mid_files("./dataset/nesmdb/nesmdb_midi")