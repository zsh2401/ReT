from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI
from symusic import Score
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm

def tokenize():
    target_file = "tokenizer30k.json"
    config = TokenizerConfig(num_velocities=32, use_chords=True, use_programs=True)
    tokenizer = REMI(config)
    files_paths = list(Path("dataset").glob("**/*.mid"))
    print(f"There are {len(files_paths)} mid files.")
    print(f"Training tokenizer and save model to {target_file}")
    tokenizer.train(vocab_size=30_000, files_paths=files_paths)
    tokenizer.save(target_file)

if __name__ == '__main__':
    tokenize()

