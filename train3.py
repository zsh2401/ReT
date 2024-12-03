# 模型参数
import torch
from torch import nn


from dataset.dataset import NesMusicDataset
from dataset.vocab import word2idx, get_vocab_size
from model import MusicTransformer
import tqdm
import sys
import datetime
from torch.utils.data import DataLoader

from model2 import MusicTransformer2

TRAIN_CODE = "v3-with-decoder-only"

# 训练循环
num_epochs = 200
batch_size = 64
seq_len = 1024
embed_dim = 512
num_heads = 8
num_layers = 6
dff = 1024
max_len = 2048  # 填充后的最大序列长度
pad_token = word2idx("<PAD>")
vocab_size = get_vocab_size()
device = "cpu"

if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

dataset = NesMusicDataset(seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
model = MusicTransformer2(vocab_size, embed_dim, num_heads, num_layers, dff).to(device)


def lr_lambda(step):
    warmup_steps = 4000  # 预热步数
    return (step / warmup_steps) if step < warmup_steps else (1 / (step**0.5))


# 忽略 PAD token 的损失
criterion = nn.CrossEntropyLoss(ignore_index=pad_token).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# scheduler = torch.optim.lr_scheduler.




losses = []
start_epoch = 0
if len(sys.argv) > 1:
    checkpoint = torch.load(sys.arv[1])
    model.load_state_dict(checkpoint["model"])
    criterion.load_state_dict(checkpoint["criterion"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    losses = checkpoint["losses"]
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"]


for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for batch_inputs, batch_targets in tqdm.tqdm(
        dataloader, desc=f"[{TRAIN_CODE}][{epoch}/{num_epochs}] Training"
    ):

        # 检查目标序列范围
        assert (
            torch.max(batch_targets) < vocab_size
        ), "Target token exceeds vocabulary size!"
        assert (
            torch.min(batch_targets) >= 0
        ), "Target token contains invalid negative index!"

        input_seq = batch_inputs.to(device).transpose(0, 1)
        target_seq = batch_targets.to(device).transpose(0, 1)

        # # 准备目标序列
        # tgt_input = batch_inputs
        # tgt_output = batch_targets

        # # print(torch.max(tgt_output), torch.min(tgt_output))
        # # 注意力掩码
        # src_padding_mask = batch_inputs == pad_token  # 忽略输入的 PAD
        # tgt_padding_mask = tgt_input == pad_token  # 忽略目标的 PAD
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(
        #     batch_inputs.device
        # )

        # 前向传播
        Y_pred = model(input_seq)
        Y_pred = Y_pred.view(-1, vocab_size)
        Y = target_seq.contiguous().view(-1)

        # print("Output logits range:", Y_pred.min().item(), Y_pred.max().item())

        # 检查模型输出
        assert Y_pred.shape[-1] == vocab_size, "Output vocabulary size mismatch!"
        assert not torch.isnan(Y_pred).any(), "Model outputs contain NaN values!"

        # 计算损失
        loss = criterion(Y_pred, Y)

        assert not torch.isnan(loss).any(), "Loss is NaN!"

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")
    losses.append(total_loss)
    torch.save(
        {
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict,
            "losses": losses,
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        },
        f"checkpoints/[{TRAIN_CODE}][{epoch}][{total_loss:.4f}][{datetime.datetime.now()}].pt",
    )
