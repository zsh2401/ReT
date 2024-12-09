from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI
from symusic import Score
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm

def tokenize():
    config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
    tokenizer = REMI(config)
    files_paths = list(Path("dataset").glob("**/*.mid"))
    print(f"There are {len(files_paths)} mid files.")
    # files_paths = tqdm.tqdm(files_paths)
    tokenizer.train(vocab_size=30000, files_paths=files_paths)
    tokenizer.save("tokenizer.json")

if __name__ == '__main__':
    tokenize()

