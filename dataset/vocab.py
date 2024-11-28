import json

from dataset.seq import TOKEN_BEGIN, TOKEN_END, TOKEN_PAD, midi_files, seq_of
import tqdm


def build_vocab():
    vocab = set()
    vocab.add(TOKEN_PAD)
    vocab.add(TOKEN_BEGIN)
    vocab.add(TOKEN_END)
    for file in tqdm.tqdm(midi_files, desc="Building vocabulary"):
        for note in seq_of(file):
            vocab.add(note)
    import json

    v_dict = {token: idx for idx, token in enumerate(sorted(vocab))}
    with open("vocab.json", "w") as f:
        json.dump(v_dict, f)

def load_vocab():
    with open("vocab.json", "r") as f:
        return json.load(f)




def vocab_size():
    word2idx("")
    return len(_in_memory_word2idx)

_in_memory_word2idx = dict()
_in_memory_idx2word = dict()
def __init():
    global _in_memory_word2idx
    if len(_in_memory_word2idx) == 0:
        _in_memory_word2idx = load_vocab()
        for k,v in _in_memory_word2idx.items():
            _in_memory_idx2word[v] = k
            
def word2idx(token):
    __init()
    return _in_memory_word2idx[token]

def idx2word(idx:int):
    __init()
    return _in_memory_idx2word[idx]
            
def translate_seq(seq):
    return [word2idx(note) for note in seq]
