from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI
from symusic import Score
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm

LMD_DATASET_PATH = "dataset/lmd"
NES_TRAIN_DATASET_PATH = "dataset/nesmdb/nesmdb_midi/nesmdb_midi/train"


def dataset_from(path: str, batch_size=64, max_seq_len=2048):
    config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
    print("Scanning files.")
    files_paths = list(Path(path).glob("**/*.mid"))
    print(f"Scanned, there are {len(files_paths)} files")
    tokenizer = REMI(config)
    print("Loading tokenizer...")
    # tokenizer.from_pretrained("tokenizer.json")
    print("Loaded into memory.")
    dataset = DatasetMIDI(
        files_paths=files_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    # tokenizer.
    collator = DataCollator(tokenizer.pad_token_id, labels_pad_idx=tokenizer["PAD_None"], copy_inputs_as_labels=True, shift_labels=True)
    dataloader = DataLoader(dataset, shuffle=True, drop_last=True, collate_fn=collator, batch_size=batch_size)
    return dataset, dataloader, len(tokenizer), tokenizer.pad_token_id
