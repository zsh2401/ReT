import json

from dataset.seq import TOKEN_BEGIN, TOKEN_END, TOKEN_PAD, get_all_seqs, get_seq_by_file_name
import tqdm
VOCAB_FILE = "vocab.json"


_in_memory_word2idx = dict()
_in_memory_idx2word = dict()

def build_vocab():
    vocab = set()
    for k,v in tqdm.tqdm(get_all_seqs().items(),desc="Building vocabulary"):
        for token in v["seq"]:
            vocab.add(token)
    
    import json

    v_dict = {token: idx+3 for idx, token in enumerate(sorted(vocab))}
    v_dict[TOKEN_PAD] = 0
    v_dict[TOKEN_BEGIN] = 1
    v_dict[TOKEN_END] = 2
    with open(VOCAB_FILE, "w") as f:
        json.dump(v_dict, f)

def load_vocab():
    with open(VOCAB_FILE, "r") as f:
        return json.load(f)

def get_vocab_size():
    __init_in_memory()
    return len(_in_memory_word2idx)

def __init_in_memory():
    global _in_memory_word2idx
    if len(_in_memory_word2idx) == 0:
        _in_memory_word2idx = load_vocab()
        for k,v in _in_memory_word2idx.items():
            _in_memory_idx2word[v] = k
            
def word2idx(token):
    __init_in_memory()
    return _in_memory_word2idx[token]

def idx2word(idx:int):
    __init_in_memory()
    return _in_memory_idx2word[idx]
            
def token_seq_to_idx_seq(seq):
    return [word2idx(note) for note in seq]

def idx_seq_to_token_seq(idx_seq):
    return [idx2word(idx) for idx in idx_seq]