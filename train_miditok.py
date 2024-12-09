# 模型参数
import torch
from torch import nn

from midiset import LMD_DATASET_PATH, NES_TRAIN_DATASET_PATH, dataset_from
import tqdm
import sys
import datetime

from model2 import MusicTransformer2

TRAIN_CODE = "v4"
# 训练循环
num_epochs = 200
batch_size = 56
embed_dim = 512
num_heads = 8
num_layers = 8
dff = 2048
max_len = 2048  # 填充后的最大序列长度
dataset, dataloader, vocab_size, pad_token = dataset_from(
    path=LMD_DATASET_PATH, max_seq_len=max_len, batch_size=batch_size
)

device = "cpu"
if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

# 初始化模型
model = MusicTransformer2(
    vocab_size, embed_dim, num_heads, num_layers, dff
).to(device)

print(vocab_size)


def lr_lambda(step):
    warmup_steps = 4000  # 预热步数
    return (step / warmup_steps) if step < warmup_steps else (1 / (step**0.5))


# 忽略 PAD token 的损失
criterion = nn.CrossEntropyLoss(ignore_index=pad_token).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)

losses = []
start_epoch = 0
if len(sys.argv) > 1:
    checkpoint = torch.load(sys.argv[1])
    model.load_state_dict(checkpoint["model"])
    criterion.load_state_dict(checkpoint["criterion"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    losses = checkpoint["losses"]
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"]

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm.tqdm(
        dataloader, desc=f"[{TRAIN_CODE}][{epoch}/{num_epochs}] Training"
    ):
        optimizer.zero_grad()
        
        # 把形状从(BatchSize, SeqLen)变成(SeqLen,BatchSize)
        X = batch["input_ids"].transpose(0, 1).to(device)
        # 同上
        Y = batch["labels"].transpose(0, 1).to(device)
        
        # assert torch.max(Y) < vocab_size, "Target token exceeds vocabulary size!"
        # # assert torch.min(Y) >= 0, "Target token contains invalid negative index!"
        # if torch.min(Y) < 0:
        #     print("minium", torch.min(Y))
        #     raise Exception("fuck")
        # mask = batch["attention_mask"].transpose(0, 1).to(device)
        
        # 形状为 (SeqLen, BatchSize,VocabSize)
        Y_pred = model(X)
        
        # print(Y.shape,Y_pred.shape)
        # Y = Y.contiguous().view(-1)
        # print(Y.shape)
        # Y
        # 检查模型输出
        assert Y_pred.shape[-1] == vocab_size, "Output vocabulary size mismatch!"
        assert not torch.isnan(Y_pred).any(), "Model outputs contain NaN values!"
        
        # 变换为(SeqLen*BatchSize)
        Y_flat = Y.reshape(-1)
        
        # 这里实际上就是铺平了的概率分布了
        # 变换为(SeqLen*BatchSize, VocabSize)
        Y_pred_flat = Y_pred.reshape(-1,Y_pred.size(-1))
        
        loss = criterion(Y_pred_flat, Y_flat)
        # 时空倒转！
        loss.backward()
        # print(X.shape,Y.shape,mask.shape,X[0][-10:],Y[0][-10:])
        optimizer.step()
        # optimizer.lr
        total_loss += loss.item()

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
        f"checkpoints/[{TRAIN_CODE}][{epoch}][{total_loss:.4f}][{datetime.datetime.now()}].pt",
    )
