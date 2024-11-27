import torch

from dataset.seq import midi_files, seq_of
from dataset.vocab import  note_to_token

S_len = 2048
def to_std_seq_vector(seq, s_len=S_len):
    arr = [note_to_token(note) for note in seq]
    arr.insert(0, note_to_token("<BOS>"))
    arr = arr[:s_len - 1]
    if len(arr) < s_len:
        arr.append(note_to_token("<EOS>"))
    while len(arr) < s_len:
        arr.append(note_to_token("<PAD>"))
    return arr


class NesMusicDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(NesMusicDataset, self).__init__()

    def __len__(self):
        return len(midi_files)

    def __getitem__(self, idx):
        seq = seq_of(midi_files[idx])
        seq = to_std_seq_vector(seq)
        input_seq = seq[:-1]
        target_seq = seq[1:]
        return torch.tensor(input_seq), torch.tensor(target_seq)