import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MusicTransformer2(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(MusicTransformer2, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._get_positional_encoding(max_len=5000, d_model=d_model)
        # self.positional_encoding = abs_positional_encoding(max_position=2047,d_model=d_model)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _get_positional_encoding(self, max_len, d_model):
        """Generates positional encoding for sequence."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, target, tgt_mask=None, memory=None):
        """
        Forward pass for the model.
        Args:
            target (torch.Tensor): Target sequence of shape (tgt_len, batch_size).
            memory (optional): Not used here, for compatibility with encoder-decoder structure.
        """
        tgt_len, batch_size = target.size()
        # print(target.shape)

        # Embedding and positional encoding
        x = self.embedding(target)
        # print(target.shape)
        x *= math.sqrt(self.d_model)
        
        # print(x.shape,self.positional_encoding.shape,self.positional_encoding[:tgt_len, :].shape)
        p = self.positional_encoding[:,:x.shape[-2], :].to(x.device)
        # print(x.shape,p.shape)
        x += p
        # print(x.shape)

        # Mask for causal attention
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(target.device)

        if memory is None:
            memory = torch.zeros(1, batch_size, self.d_model, device=x.device)
            
        # Decode
        decoder_output = self.decoder(x, memory=memory,tgt_mask=tgt_mask)

        # print(decoder_output)
        # Final output projection
        output = self.fc_out(decoder_output)
        return output
    
    
def abs_positional_encoding(max_position, d_model, n=3):
    """
    Since the transformer does not use recurrence or convolution, we have to deliberately give it positional
    information. Though learned relative position embeddings will be added to the model, it is possible that absolute
    position encoding will aid it in predicting next tokens.

    Args:
        max_position (int): maximum position for which to calculate positional encoding
        d_model (int): Transformer hidden dimension size
        n (int): number of dimensions to which to broadcast output

    Returns:
        sinusoidal absolute positional encoding of shape d_model for max_position positions
    """
    # set of all positions to consider
    positions = torch.arange(max_position).float()

    # get angles to input to sinusoid functions
    k = torch.arange(d_model).float()
    coeffs = 1 / torch.pow(10000, 2 * (k // 2) / d_model)
    angles = positions.view(-1, 1) @ coeffs.view(1, -1)

    # apply sin to the even indices of angles along the last axis
    angles[:, 0::2] = torch.sin(angles[:, 0::2])

    # apply cos to the odd indices of angles along the last axis
    angles[:, 1::2] = torch.cos(angles[:, 1::2])

    return angles.view(*[1 for _ in range(n-2)], max_position, d_model)