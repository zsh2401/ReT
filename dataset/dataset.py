import torch

from dataset.seq import midi_files, pad_seq, seq_of
from dataset.vocab import translate_seq


class NesMusicDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(NesMusicDataset, self).__init__()
        self.cache = {}

    def __len__(self):
        return len(midi_files)

    def __getitem__(self, idx):
        seq = seq_of(midi_files[idx])
        seq = pad_seq(seq)
        seq = translate_seq(seq)
        # print(seq)
        input_seq = seq[:-1]
        target_seq = seq[1:]
        return torch.tensor(input_seq), torch.tensor(target_seq)
