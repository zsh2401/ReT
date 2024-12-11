import argparse

import torch
from t3.midiset import tokenizer_
from t3.model2 import MusicTransformer2
import pretty_midi
from pydub import AudioSegment
# 保存为 WAV 文件
import numpy as np
from scipy.io.wavfile import write

# def gener

def main():
    tokenizer = tokenizer_()
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-point",required=True, type=str)
    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--num-layers", default=8, type=int)
    parser.add_argument("--dff", default=2048, type=int)
    parser.add_argument("--max-len", default=2048, type=int)
    parser.add_argument("--vocab-size", type=int,default=len(tokenizer))
    args = parser.parse_args()
    
    device = "cpu"
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        
    model =  MusicTransformer2(args.vocab_size, args.embed_dim, args.num_heads, args.num_layers, args.dff).to(
        device
    )
    checkpoint = torch.load(args.check_point,map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    
    eos_token_id = tokenizer["BOS_None"]
    X = torch.tensor([[tokenizer["BOS_None"]]], dtype=torch.long).to(device)
    max_length = 2048
    with torch.no_grad():
        for _ in range(max_length - X.size(-1)):
            Y = model(X)
            next_token_logits = Y[:,-1,:]
            next_token_id = top_k_sampling(next_token_logits,5)
            # print(next_token_id)
            # print(X.shape,next_token_id.shape)
            # print(next_token_id)
            X = torch.cat([X, next_token_id],dim=-1)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
    
    # SEQ = X[0].tolist()
    bos = tokenizer["BOS_None"]
    eos = tokenizer["EOS_None"]
    pad = tokenizer.pad_token_id
    special_tokens = {bos,eos,pad}
    SEQ = X[0].tolist()
    SEQ = list(filter(lambda token: token not in special_tokens, SEQ))
    print(SEQ)
    midi = tokenizer(SEQ)
    midi.dump_midi("a.mid")
    # 加载 MIDI 文件
    midi_data = pretty_midi.PrettyMIDI('a.mid')

    # 将 MIDI 转换为音频 (synthesized)
    audio_data = midi_data.synthesize()

    sample_rate = 44100  # 采样率
    write("output.wav", sample_rate, np.int16(audio_data * 32767))  # 保存 WAV
    print("Conversion complete: output.wav")

def top_k_sampling(logits, k=5):
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probabilities = torch.softmax(top_k_logits, dim=-1)
    sampled_index = torch.multinomial(probabilities, 1)
    return top_k_indices[0, sampled_index]

if __name__ == "__main__":
    main()