import torch

from dataset.seq import midi_files, seq_of
from dataset.vocab import V_dict

S_len = 2048
def to_std_seq_vector(seq, s_len=S_len):
    arr = [V_dict[a] for a in seq]
    arr.insert(0, V_dict["<BOS>"])
    arr = arr[:s_len - 1]
    if len(arr) < s_len:
        arr.append(V_dict["<EOS>"])
    while len(arr) < s_len:
        arr.append(V_dict["<PAD>"])
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