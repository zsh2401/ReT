
from dataset.seq import build_seq, load_seq, pad_seq
# from dataset.tokenization import time_to_tokens
from dataset.vocab import build_vocab
import tqdm

# for i in tqdm.tqdm(range(3000),desc="Heating time to tokens algorithms"):
#     time_to_tokens(i)
# for i in tqdm.tqdm(range(3000),desc="Testing time to tokens cache"):
#     time_to_tokens(i) 
build_seq()
build_vocab()

