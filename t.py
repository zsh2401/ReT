from dataset.dataset import NesMusicDataset
from dataset.seq import pad_seq
from dataset.vocab import idx2word, token_seq_to_idx_seq, word2idx

S = pad_seq(["Control_11_1"],10)
print(S)
print(token_seq_to_idx_seq(S))
print(idx2word(0))
print(word2idx("<PAD>"))
# dataset = NesMusicDataset()
# print(dataset[0])