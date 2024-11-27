import json

from dataset.seq import midi_files, seq_of
import tqdm

def build_vocab():
    vocab = set()
    vocab.add("<PAD>")
    vocab.add("<BOS>")
    vocab.add("<EOS>")
    for file in tqdm.tqdm(midi_files, desc="Building vocabulary"):
        for note in seq_of(file):
            vocab.add(note)
    import json
    V_dict = {token: idx for idx, token in enumerate(sorted(vocab))}
    with open("vocab.json", "w") as f:
        json.dump(V_dict, f)


def load_vocab():
    with open("vocab.json", "r") as f:
        return json.load(f)


build_vocab()
V_dict = load_vocab()