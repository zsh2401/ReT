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
    midi = MidiFile(midi_path)
    from dataset.tokenization import tokenize
    return tokenize(midi)

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
