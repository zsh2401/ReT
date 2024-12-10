from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI
from symusic import Score
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm
import random
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler

from t3.dataset import MDataset

LMD_DATASET_PATH = "dataset/lmd"
NES_TRAIN_DATASET_PATH = "dataset/nesmdb/nesmdb_midi/nesmdb_midi/train"


def random_pick(
    paths: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    # 检查比例是否满足总和为 1
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError(
            "train_ratio, val_ratio, and test_ratio must be between 0 and 1"
        )
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1")

    # 随机打乱路径列表
    random.shuffle(paths)

    # 按比例计算分割点
    total_count = len(paths)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    # 划分数据集
    train_set = paths[:train_end]
    val_set = paths[train_end:val_end]
    test_set = paths[val_end:]

    return train_set, val_set, test_set

def tokenizer_():
    config = TokenizerConfig(num_velocities=64, use_chords=True, use_programs=True)
    return REMI(config)

def dataset_from(path: str,
                 batch_size=64,
                 max_seq_len=2048,
                 ddp=False,
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1,
                    test_ratio: float = 0.1,
                 ):
    config = TokenizerConfig(num_velocities=64, use_chords=True, use_programs=True)
    print("Scanning files.")
    files_paths = list(Path(path).glob("**/*.mid"))
    print(f"Scanned, there are {len(files_paths)} files")
    tokenizer = REMI(config)
    print("Loading tokenizer...")
    # tokenizer.from_pretrained("tokenizer.json")
    print("Loaded into memory.")
    train_files, val_files, test_files = random_pick(files_paths,train_ratio,val_ratio,test_ratio)

    train_dataset = MDataset(train_files,tokenizer)
    test_dataset = MDataset(test_files, tokenizer)
    val_dataset = MDataset(val_files, tokenizer)
    # train_dataset = DatasetMIDI(
    #     files_paths=train_files,
    #     tokenizer=tokenizer,
    #     max_seq_len=max_seq_len,
    #     bos_token_id=tokenizer["BOS_None"],
    #     eos_token_id=tokenizer["EOS_None"],
    # )
    # val_dataset = DatasetMIDI(
    #     files_paths=val_files,
    #     tokenizer=tokenizer,
    #     max_seq_len=max_seq_len,
    #     bos_token_id=tokenizer["BOS_None"],
    #     eos_token_id=tokenizer["EOS_None"],
    # )
    # test_dataset = DatasetMIDI(
    #     files_paths=test_files,
    #     tokenizer=tokenizer,
    #     max_seq_len=max_seq_len,
    #     bos_token_id=tokenizer["BOS_None"],
    #     eos_token_id=tokenizer["EOS_None"],
    # )

    # tokenizer.
    # collator = DataCollator(
    #     tokenizer.pad_token_id,
    #     labels_pad_idx=tokenizer["PAD_None"],
    #     copy_inputs_as_labels=True,
    #     shift_labels=True,
    # )
    
    train_sampler = DistributedSampler(dataset=train_dataset) if ddp else RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        pin_memory=ddp,
        # drop_last=False,
        batch_size=batch_size,
    )

    val_sampler = DistributedSampler(dataset=val_dataset) if ddp else RandomSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        pin_memory=ddp,
        # drop_last=True,
        batch_size=batch_size,
    )

    test_sampler = DistributedSampler(dataset=test_dataset) if ddp else RandomSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        pin_memory=ddp,
        # drop_last=True,
        batch_size=batch_size,
    )

    return (
        train_sampler,
        train_dataset,
        train_dataloader,
        val_sampler,
        val_dataset,
        val_dataloader,
        test_sampler,
        test_dataset,
        test_dataloader,
        len(tokenizer),
        tokenizer.pad_token_id,
        tokenizer["BOS_None"],
        tokenizer["EOS_None"],
    )
