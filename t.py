from dataset.dataset import NesMusicDataset
from dataset.seq import pad_seq
from dataset.vocab import translate_seq

# S = pad_seq(["Control_11_1"],3)
# print(S)
# print(translate_seq(S))
dataset = NesMusicDataset()
print(dataset[0])