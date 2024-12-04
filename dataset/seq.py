import os
from mido import MidiFile
import json
import tqdm
from dataset.tokenization import tokenize, __time_to_tokens_cache
from dataset.tokenization2 import tokenize2

FILE_NAME = "seq.json"
# PATH_TO_MIDI = "./dataset/lmd"
PATH_TO_MIDI = "./dataset/nesmdb/nesmdb_midi/nesmdb_midi/test"
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


def tokenize_mid_file(midi_path):
    return tokenize2(midi_path)
    # midi = MidiFile(midi_path)
    # return tokenize(midi, shared_ttt_cache)


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


def __midi_preprocess_worker(chunk, result_queue):
    import os
    import time
    for _file in chunk:
        try:
            start = time.time()
            tokens = tokenize_mid_file(_file)
            end = time.time()
            # print(f"{os.getpid()} used {end - start}s")
            result_queue.put(("success", os.getpid(), _file, {"seq": tokens}))
        except Exception as e:
            result_queue.put(("error", os.getpid(), f"{e}"))


def build_seq():
    """
    构建序列化文件
    """
    import os

    import numpy as np

    # import threading
    from multiprocessing import Process, Manager, Pipe
    import psutil
    import math

    process_count = math.ceil(psutil.cpu_count(logical=True) * 1)
    midi_files = find_all_mid_files(PATH_TO_MIDI)
    midi_files_chunks = np.array_split(midi_files, process_count)
    # print(len(mi))

    final_seq_data = dict()
    with Manager() as manager:
        result_queue = manager.Queue()
        processes = []

        for chunk in midi_files_chunks:
            p = Process(
                target=__midi_preprocess_worker,
                args=(chunk, result_queue),
            )
            processes.append(p)
            p.start()

        with tqdm.tqdm(
            total=len(midi_files),
            desc=f"Building sequences with {process_count} threads.",
        ) as progress_bar:
            completed = 0
            while completed < len(midi_files):
                task_result = result_queue.get()
                # print(f"message from {task_result[1]}")
                if task_result[0] == "success":
                    final_seq_data[task_result[2]] = task_result[3]
                else:
                    pass
                completed += 1
                progress_bar.update(1)

        for p in processes:
            p.join()

        with open(FILE_NAME, "w") as f:
            json.dump(final_seq_data, f)


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
