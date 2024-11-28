# 模型参数
import torch
from torch import nn

from dataset.dataset import NesMusicDataset
from dataset.vocab import  word2idx,vocab_size
from model import MusicTransformer
import tqdm
import sys
import datetime

TRAIN_CODE = "v1024"
# 训练循环
num_epochs = 200
batch_size = 24
embed_dim = 64
num_heads = 4
num_layers = 6
dff = 2048
max_len = 2048  # 填充后的最大序列长度
pad_token = word2idx("<PAD>")
vocab_size = vocab_size()
device = "cpu"
if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

# 初始化模型
model = MusicTransformer(vocab_size, embed_dim, num_heads, num_layers, max_len, pad_token,dff).to(device)

# 忽略 PAD token 的损失
criterion = nn.CrossEntropyLoss(ignore_index=pad_token).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
from torch.utils.data import DataLoader

losses = []
start_epoch = 0
if len(sys.argv) > 1:
    checkpoint = torch.load(sys.arv[1])
    model.load_state_dict(checkpoint["model"])
    criterion.load_state_dict(checkpoint["criterion"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    losses = checkpoint["losses"]
    start_epoch = checkpoint["epoch"]
    
dataset = NesMusicDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(start_epoch,num_epochs):
    model.train()
    total_loss = 0
    for batch_inputs, batch_targets in tqdm.tqdm(dataloader, desc=f"[{TRAIN_CODE}][{epoch}/{num_epochs}] Training"):
        optimizer.zero_grad()

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        # 准备目标序列
        tgt_input = batch_inputs
        tgt_output = batch_targets

        # 注意力掩码
        src_padding_mask = (batch_inputs == pad_token)  # 忽略输入的 PAD
        tgt_padding_mask = (tgt_input == pad_token)    # 忽略目标的 PAD
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(batch_inputs.device)

        # 前向传播
        outputs = model(batch_inputs, tgt_input, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)

        # 计算损失
        loss = criterion(outputs.view(-1, vocab_size), tgt_output.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")
    losses.append(total_loss)
    torch.save({
        "model":model.state_dict(),
        "criterion":criterion.state_dict(),
        "optimizer":optimizer.state_dict,
        "losses":losses,
        "epoch":epoch
    },f"checkpoints/[{TRAIN_CODE}][{epoch}][{total_loss:.4f}][{datetime.datetime.now()}].pt")