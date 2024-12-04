import torch

from dataset.seq import all_seq_size, midi_files, pad_seq, get_seq_by_file_name
from dataset.vocab import idx_seq_to_token_seq, token_seq_to_idx_seq


class NesMusicDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len: int=2048):
        super(NesMusicDataset, self).__init__()
        self.cache = {}
        self.seq_len = seq_len

    def __len__(self):
        return all_seq_size()

    def __getitem__(self, idx):
        
        seq = get_seq_by_file_name(midi_files[idx])
        seq = pad_seq(seq, self.seq_len)
        seq = token_seq_to_idx_seq(seq)
        
        # print(seq)
        input_seq = seq[:-1]
        target_seq = seq[1:]
        # print(idx_seq_to_word_seq(input_seq))
        # print(idx_seq_to_word_seq(target_seq))
        # raise ""
        return torch.tensor(input_seq), torch.tensor(target_seq)
