from pathlib import Path
from typing import Sequence
from miditok import MusicTokenizer
import torch
from miditok.pytorch_data import DatasetMIDI


class MDataset(torch.utils.data.Dataset):

    def __init__(self, midi_files: list[str],
                 _tk: MusicTokenizer,
                 seq_len: int = 2048,
                 sliding_step:int = None):
        super().__init__()
        self.source = DatasetMIDI(
            files_paths=midi_files,
            tokenizer=_tk,
            max_seq_len=-1,
            bos_token_id=_tk["BOS_None"],
            eos_token_id=_tk["EOS_None"],
        )
        self.sliding_step = sliding_step if sliding_step is not None else seq_len // 2
        self.next_step = {}
        self.seq_len = seq_len
        self.cache = dict[int, Sequence[float]]()

    def __len__(self):
        return len(self.source)

    def pad_to_multiple(self, arr: list[any]):
        """
        如果数组长度不是指定倍数，填充特定数量的值。
        Args:
            arr: list[int]，输入数组
            multiple: int，目标倍数
            pad_value: int，填充值
        Returns:
            list[int]，处理后的数组
        """
        length = len(arr)
        mask = [1] * length
        remainder = length % self.seq_len  # 计算当前长度是否满足倍数
        if remainder != 0:
            padding = self.seq_len - remainder  # 需要填充的数量
            arr.extend([self.source.tokenizer.pad_token_id] * padding)  # 填充
            mask.extend([0] * padding)  # 填充部分 mask 为 0
        return arr, mask

    def __getitem__(self, index):
        import numpy as np
        if index not in self.cache:
            raw = self.source[index]
            tensor = raw["input_ids"]
            if tensor is None:
                tensor = torch.zeros(10)
            seq = tensor.tolist()
            seq = seq + [self.source.eos_token_id]
            self.cache[index] = np.array(seq)
        else:
            seq = self.cache[index]

        if index not in self.next_step:
            self.next_step[index] = 0

        step = self.next_step[index]
        if step * self.sliding_step + self.seq_len >= len(seq):
            self.next_step[index] = 0
        else:
            self.next_step[index] += 1

        raw_seq = list(seq[step * self.sliding_step:][:self.seq_len])
        pad_seq, mask = self.pad_to_multiple(raw_seq)
        X = pad_seq[:-1]
        X_mask = mask[:-1]
        Y = pad_seq[1:]
        Y_mask = mask[1:]
        return torch.tensor(X), torch.tensor(Y), torch.tensor(X_mask), torch.tensor(Y_mask)
