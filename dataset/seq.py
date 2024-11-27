import os
from mido import MidiFile

import tqdm

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



_raw_sequence = dict()

def seq_of(file):
    if len(_raw_sequence) == 0:
        for _file in tqdm.tqdm(midi_files, desc="Building sequences"):
            _raw_sequence[_file] = midi_to_sequence(_file)
    return _raw_sequence[file]

midi_files = find_all_mid_files("./dataset/nesmdb/nesmdb_midi")