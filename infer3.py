import sys
import torch
from dataset.vocab import get_vocab_size, idx_seq_to_token_seq, word2idx
from model2 import MusicTransformer2  # 假设你的模型定义在 model.py 中

# from midi_utils import save_midi  # 假设你有一个 save_midi 工具函数将序列保存为 MIDI


# 定义推理函数
def generate_sequence(model, start_token, max_len=100, eos_token=None, device="cpu"):
    """
    生成 MIDI 序列。

    Args:
        model: 训练好的 MusicTransformer 模型。
        start_token: 生成的起始标记 (int)。
        max_len: 生成序列的最大长度 (int)。
        eos_token: 序列结束标记 (int)，遇到时停止生成。
        device: 推理运行的设备 ('cpu' or 'cuda')。

    Returns:
        generated_sequence: 生成的 MIDI token 序列 (list of int)。
    """
    model.eval()
    generated_sequence = [start_token]

    with torch.no_grad():
        for _ in range(max_len):
            # 构造当前输入序列
            input_seq = torch.tensor(
                generated_sequence, dtype=torch.long, device=device
            ).unsqueeze(
                1
            )  # (seq_len, 1)

            # 生成目标序列掩码
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                input_seq.size(0)
            ).to(device)

            # 前向推理
            output = model(input_seq, memory=None)

            # 获取最后一个时间步的输出，预测下一个 token
            next_token = torch.argmax(output[-1, 0]).item()

            # 如果遇到 eos_token，停止生成
            if eos_token is not None and next_token == eos_token:
                break

            # 将预测的 token 添加到序列中
            generated_sequence.append(next_token)

    return generated_sequence


# 主程序
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    # 配置
    vocab_size = get_vocab_size()  # 根据训练时的词汇表大小调整
    d_model = 256  # 根据训练时的模型参数调整
    num_heads = 8
    num_layers = 6
    dff = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = MusicTransformer2(vocab_size, d_model, num_heads, num_layers, dff)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print(f"Loaded model checkpoint from {checkpoint_path}")

    # 起始标记 (BOS)
    start_token = word2idx("<BOS>")  # 根据训练时定义的 BOS token
    eos_token = word2idx("<EOS>")  # 根据训练时定义的 EOS token
    max_len = 2048  # 最大生成长度

    # 生成 MIDI 序列
    generated_sequence = generate_sequence(
        model, start_token, max_len=max_len, eos_token=eos_token, device=device
    )
    
    print("Generated sequence:", idx_seq_to_token_seq(generated_sequence))

    # 保存为 MIDI 文件
    # output_midi_path = "generated_music.mid"
    # save_midi(generated_sequence, output_midi_path)
    # print(f"Generated MIDI file saved to {output_midi_path}")
