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


def midi_to_token_seq(midi_path):
    midi = MidiFile(midi_path)
    from dataset.tokenization import tokenize

    return tokenize(midi)


def get_seq_by_file_name(file):
    """
    获取没有经过对齐的，纯字符格式的序列
    """
    return get_all_seqs()[file]["seq"]


__in_memory_seq = None

def get_all_seqs():
    global __in_memory_seq
    if __in_memory_seq is None:
        __in_memory_seq = load_seq()
    return __in_memory_seq


def calculate_avg_len():
    sum = 0
    for seq in get_all_seqs():
        sum += len(seq)
        
def all_seq_size():
    return len(get_all_seqs())


def load_seq():
    with open(FILE_NAME, "r") as f:
        return json.load(f)


def __midi_preprocess_worker(shared_data_dict, chunk, signal_queue):
    for _file in chunk:
        try:
            seq = midi_to_token_seq(_file)
            shared_data_dict[_file] = {
                "seq": seq,
            }
        except Exception as e:
            pass
        finally:
            signal_queue.put(1)


def build_seq():
    """ """
    import os
    import numpy as np

    # import threading
    from multiprocessing import Process, Manager, Pipe
    import psutil
    import math

    process_count = math.ceil(psutil.cpu_count(logical=True) * 1)

    midi_files = find_all_mid_files(PATH_TO_MIDI)
    midi_files_chunks = np.array_split(midi_files, process_count)

    with Manager() as manager:
        signal_queue = manager.Queue()
        shared_dict = manager.dict()
        processes = []

        for chunk in midi_files_chunks:
            p = Process(
                target=__midi_preprocess_worker,
                args=(shared_dict, chunk, signal_queue),
            )
            processes.append(p)
            p.start()

        with tqdm.tqdm(
            total=len(midi_files),
            desc=f"Building sequences with {process_count} threads.",
        ) as progress_bar:
            completed = 0
            while completed < len(midi_files):
                sig = signal_queue.get()
                completed += sig
                progress_bar.update(sig)

        for p in processes:
            p.join()

        with open(FILE_NAME, "w") as f:
            json.dump(shared_dict, f)


def pad_seq(raw, s_len: int = S_len):
    """
    将序列进行对齐
    """
    copy = [token for token in raw]
    copy.insert(0, TOKEN_BEGIN)
    copy = copy[: s_len - 1]
    if len(copy) < s_len:
        copy.append(TOKEN_END)
    while len(copy) < s_len:
        copy.append(TOKEN_PAD)
    return copy
