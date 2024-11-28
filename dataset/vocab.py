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
    token_to_idx("")
    return len(_in_memory_dict)

_in_memory_dict = dict()
def token_to_idx(note):
    global _in_memory_dict
    if len(_in_memory_dict) == 0:
        _in_memory_dict = load_vocab()
    return _in_memory_dict[note]

def translate_seq(seq):
    return [token_to_idx(note) for note in seq]
