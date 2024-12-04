from miditok import REMI, TokenizerConfig
from symusic import Score

# Creating a multitrack tokenizer, read the doc to explore all the parameters
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)

def tokenize2(midi_file_path):
    midi = Score(midi_file_path)
    tokens = tokenizer(midi)  # calling the tokenizer will automatically detect MIDIs, paths and tokens
    # converted_back_midi = tokenizer(tokens) 
    return tokens