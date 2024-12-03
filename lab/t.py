import torch
import torch.nn as nn

# 假设目标序列形状 (tgt_len, batch_size, d_model)
tgt_len, batch_size, d_model = 10, 2, 512
x = torch.rand(tgt_len, batch_size, d_model)

# 定义 Decoder Layer 和 Decoder
decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
# 
# if memory is None:
memory = torch.zeros(1, batch_size, d_model, device=x.device)
    
# Memory 为 None
tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len)
output = decoder(x, memory=None, tgt_mask=tgt_mask)

print("Output shape:", output.shape)

# midi_to_sequence