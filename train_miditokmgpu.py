# 模型参数
import torch
from torch import nn
from midiset import LMD_DATASET_PATH, NES_TRAIN_DATASET_PATH, dataset_from
import tqdm
import sys
import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data.distributed import DistributedSampler
from model2 import MusicTransformer2
import os
import argparse

smoothing = SmoothingFunction().method1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", default=os.getenv("LOCAL_RANK", -1), type=int)
    parser.add_argument("--check-point", default=None, type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--dataset", default=LMD_DATASET_PATH, type=str)
    parser.add_argument("--d-model", default=512, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--num-layers", default=8, type=int)
    parser.add_argument("--dff", default=2048, type=int)
    parser.add_argument("--max-len", default=2048, type=int)
    parser.add_argument("--lr", default="1e-4", type=str)
    parser.add_argument("--weight-decay", default="1e-4", type=str)
    parser.add_argument("--train-code", default="Unk", type=str)
    parser.add_argument("--mode", default="train", type=str)
    args = parser.parse_args()
    print(args)

    train_code = args.train_code
    num_epochs = 200
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    dff = args.dff
    max_len = args.max_len  # 填充后的最大序列长度
    local_rank = args.local_rank
    batch_size = args.batch_size
    print(num_heads, num_layers)
    print(f"Local Rank: {local_rank}")

    device = "cpu"
    if args.local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    elif torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    (
        train_sampler,
        train_dataset,
        train_dataloader,
        val_sampler,
        val_dataset,
        val_dataloader,
        test_sampler,
        test_dataset,
        test_dataloader,
        vocab_size,
        pad_token,
        bos_token,
        eos_token
    ) = dataset_from(
        path=args.dataset, max_seq_len=max_len, batch_size=batch_size, ddp=args.local_rank != -1
    )

    # 初始化模型
    model = MusicTransformer2(vocab_size, d_model, num_heads, num_layers, dff).to(
        device
    )

    if args.local_rank != -1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # 忽略 PAD token 的损失
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
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

    def train(epoch: int):

        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        model.train()
        train_losses = []
        with tqdm.tqdm(
                total=len(train_dataloader),
                desc=f"[{train_code}][{epoch}/{num_epochs}] Training"
        ) as bar:
            # batch_num
            a = iter(train_dataloader)
            while True:
                try:
                    # X,Y,X_mask,Y_mask: BatchSize, SeqLen
                    X, Y, X_mask, Y_mask = next(a)
                except StopIteration:
                    break
                except torch.OutOfMemoryError as e:
                    raise e
                except Exception as e:
                    print(e)

                optimizer.zero_grad()

                # 把形状从(BatchSize, SeqLen)变成(SeqLen,BatchSize)
                X = X.transpose(0, 1).to(device)

                # 同上
                Y = Y.transpose(0, 1).to(device)

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

                train_losses.append(loss.item())

                bar.set_description(f"[{train_code}][{epoch}/{num_epochs}] Trained, batch loss={loss.item():.5f}")

                bar.update(1)

        scheduler.step()
        avg_loss = sum(train_losses) / len(train_losses)
        return avg_loss

    def validate(epoch: int):
        if isinstance(val_sampler, DistributedSampler):
            val_sampler.set_epoch(epoch)

        def remove_special_tokens(sequence):
            return [token for token in sequence if token != pad_token and token != eos_token]

        model.eval()
        val_bleus = []

        val_losses = []
        perplexities = []

        with torch.no_grad():
            with tqdm.tqdm(
                    total=len(val_dataloader), desc=f"[{train_code}][{epoch}/{num_epochs}] Validating"
            ) as bar:
                # batch_num
                val_iter = iter(val_dataloader)
                while True:
                    try:
                        # X,Y,X_mask,Y_mask: BatchSize, SeqLen
                        X, Y, X_mask, Y_mask = next(a)
                    except StopIteration:
                        break
                    except torch.OutOfMemoryError as e:
                        raise e
                    except Exception as e:
                        print(e)

                    # 把形状从(BatchSize, SeqLen)变成(SeqLen,BatchSize)
                    X = X.transpose(0, 1).to(device)

                    # 把形状从(BatchSize, SeqLen)变成(SeqLen,BatchSize)
                    Y = Y.transpose(0, 1).to(device)

                    # 形状为 (SeqLen, BatchSize, VocabSize)
                    Y_pred = model(X)

                    Y_pred_tokens = Y_pred.argmax(dim=-1)
                    Y_pred_tokens = Y_pred_tokens.transpose(0, 1).tolist()  # (SeqLen, BatchSize) -> (BatchSize, SeqLen)
                    Y_tokens = Y.transpose(0, 1).tolist()  # (SeqLen, BatchSize) -> (BatchSize, SeqLen)

                    bleu_scores = []
                    for Y_a, Y_b in zip(Y_tokens, Y_pred_tokens):
                        Y_a = remove_special_tokens(Y_a)
                        Y_b = remove_special_tokens(Y_b)
                        bleu = sentence_bleu([Y_a], Y_b, smoothing_function=smoothing)
                        bleu_scores.append(bleu)

                    val_bleus.append(sum(bleu_scores) / len(bleu_scores))

                    # 变换为(SeqLen*BatchSize)
                    Y_flat = Y.reshape(-1)

                    # 这里实际上就是铺平了的概率分布了
                    # 变换为(SeqLen*BatchSize, VocabSize)
                    Y_pred_flat = Y_pred.reshape(-1, Y_pred.size(-1))

                    # 计算loss
                    loss = criterion(Y_pred_flat, Y_flat)

                    val_losses.append(loss.item())
                    perplexities.append(torch.exp(loss))

                    bar.update(1)

            avg_bleu = sum(val_bleus) / len(val_bleus)
            avg_ppl = sum(perplexities) / len(perplexities)
            avg_loss = sum(val_losses) / len(val_losses)

        return avg_ppl, avg_loss, avg_bleu

    for epoch in range(start_epoch, num_epochs):
        train_loss = train(epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}")
        ppl, val_loss, bleu = validate(epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, PPL:{ppl:.6f}, Val Loss: {val_loss:.6f}, Bleu: {bleu:.6f}")
        losses.append(train_loss)
        torch.save(
            {
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "losses": losses,
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            f"checkpoints/[{train_code}][{epoch}][{val_loss:.4f}].pt",
        )


if __name__ == "__main__":
    main()
