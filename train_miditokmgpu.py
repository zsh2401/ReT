# 模型参数
import torch
from torch import nn

from midiset import LMD_DATASET_PATH, NES_TRAIN_DATASET_PATH, dataset_from
import tqdm
import sys
import datetime

from model2 import MusicTransformer2
import os
import argparse


# 训练循环



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", default=os.getenv("LOCAL_RANK", -1), type=int)
    parser.add_argument("--check-point", default=None, type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--num-layers", default=8, type=int)
    parser.add_argument("--dff", default=8, type=int)
    parser.add_argument("--max-len", default=2048, type=int)
    parser.add_argument("--lr", default=1e-4, type=int)
    parser.add_argument("--weight-decay", default=1e-4, type=int)
    parser.add_argument("--train-code", default="Unk", type=str)
    args = parser.parse_args()
    print(args)

    train_code = args.train_code
    num_epochs = 200
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    num_layers = args.num_layers
    dff = args.dff
    max_len = args.max_len  # 填充后的最大序列长度
    local_rank = args.local_rank
    batch_size = args.batch_size
    print(num_heads,num_layers)
    print(f"Local Rank: {local_rank}")

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 1, "Must using more than single GPU"

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    dataset, dataloader, vocab_size, pad_token, sampler = dataset_from(
        path=LMD_DATASET_PATH, max_seq_len=max_len, batch_size=batch_size, ddp=True
    )

    # 初始化模型
    model = MusicTransformer2(vocab_size, embed_dim, num_heads, num_layers, dff).to(
        device
    )

    print("use {} gpus!".format(num_gpus))
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # 忽略 PAD token 的损失
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    losses = []
    start_epoch = 0
    if args.check_point is not None:
        checkpoint = torch.load(args.check_point)
        model.load_state_dict(checkpoint["model"])
        criterion.load_state_dict(checkpoint["criterion"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        losses = checkpoint["losses"]
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]

    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        with tqdm.tqdm(total=len(dataloader), desc=f"[{train_code}][{epoch}/{num_epochs}] Training") as bar:
            a = iter(dataloader)
            while True:
                try:
                    batch = next(a)
                    optimizer.zero_grad()

                    # 把形状从(BatchSize, SeqLen)变成(SeqLen,BatchSize)
                    X = batch["input_ids"].transpose(0, 1).to(device)
                    
                    # 同上
                    Y = batch["labels"].transpose(0, 1).to(device)

                    # 形状为 (SeqLen, BatchSize,VocabSize)
                    Y_pred = model(X)

                    # 变换为(SeqLen*BatchSize)
                    Y_flat = Y.reshape(-1)

                    # 这里实际上就是铺平了的概率分布了
                    # 变换为(SeqLen*BatchSize, VocabSize)
                    Y_pred_flat = Y_pred.reshape(-1, Y_pred.size(-1))

                    loss = criterion(Y_pred_flat, Y_flat)
                    # 时空倒转！
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                except ValueError as e:
                    print(e)
                    continue
                except StopIteration:
                    break;
                finally:
                    bar.update(1)

        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")
        losses.append(total_loss)
        torch.save(
            {
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "losses": losses,
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            f"checkpoints/[{train_code}][{epoch}][{total_loss:.4f}][{datetime.datetime.now()}].pt",
        )
        
if __name__ == "__main__":
    main()