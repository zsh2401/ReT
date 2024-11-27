import torch
import torch.nn as nn


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len, pad_token):
        super(MusicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token)
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, embed_dim))
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.pad_token = pad_token

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # Embedding + Positional Encoding
        src_emb = self.embedding(src) + self.positional_encoding[:src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:tgt.size(1), :]

        # Transformer
        output = self.transformer(
            src_emb.permute(1, 0, 2),  # 转换为 [seq_len, batch_size, embed_dim]
            tgt_emb.permute(1, 0, 2),  # 转换为 [seq_len, batch_size, embed_dim]
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.fc_out(output.permute(1, 0, 2))  # 转回 [batch_size, seq_len, vocab_size]