import torch
from torch import nn

from dataset.dataset import NesMusicDataset
from dataset.vocab import  word2idx,get_vocab_size
from model import MusicTransformer
import tqdm
import sys
import datetime
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def train_ddp(rank, world_size, config):
    # 初始化分布式进程组
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 配置
    vocab_size = config['vocab_size']
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    num_layers = config['num_layers']
    max_len = config['max_len']
    pad_token = config['pad_token']
    epochs = config['epochs']

    # 模拟数据集
    dataset = NesMusicDataset() # 输出序列
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

    # 初始化模型
    model = MusicTransformer(vocab_size, embed_dim, num_heads, num_layers, max_len, pad_token).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    optimizer = nn.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 确保每个 epoch 数据不同
        model.train()
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(rank), tgt.to(rank)

            # 目标序列
            tgt_input = tgt[:, :-1]  # 输入部分
            tgt_output = tgt[:, 1:]  # 输出部分

            # 因果掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(rank)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(src, tgt_input, tgt_mask=tgt_mask)

            # 计算损失
            loss = criterion(outputs.view(-1, vocab_size), tgt_output.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if rank == 0:
            print(f"Rank {rank}, Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # 销毁进程组
    dist.destroy_process_group()
    
    
# 主函数
if __name__ == "__main__":
    import os
    from torch.multiprocessing import spawn

    # 配置
    config = {
        'vocab_size': 5000,
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 6,
        'max_len': 2048,
        'pad_token': word2idx("<PAD>"),
        'epochs': 100
    }

    # 分布式训练设置
    world_size = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "24053"

    # 启动多 GPU 训练
    spawn(train_ddp, args=(world_size, config), nprocs=world_size)