def msg_to_token(single_msg):
    msg_tokens = []
    if single_msg.time > 0:
        msg_tokens.append(f"Wait_{single_msg.time}")
    if single_msg.is_meta:
        pass
    elif single_msg.type == "note_on" and single_msg.velocity > 0:
        msg_tokens.append(f"Note_On_{single_msg.note}")
        msg_tokens.append(f"Velocity_{single_msg.velocity}")
    elif single_msg.type == "note_off" or (
            single_msg.type == "note_on" and single_msg.velocity == 0
    ):
        msg_tokens.append(f"Note_Off_{single_msg.note}")
    elif single_msg.type == "control_change":
        msg_tokens.append(f"Control_{single_msg.control}_{single_msg.value}")
    elif single_msg.type == "program_change":
        msg_tokens.append(f"Program_{single_msg.program}")
    elif single_msg.type == "time_signature":
        msg_tokens.append(f"Time_Signature_{single_msg.numerator}/{single_msg.denominator}")
    elif single_msg.type == "set_tempo":
        msg_tokens.append(f"Tempo_{single_msg.tempo}")
    elif single_msg.type == "sysex":
        # 跳过sysex
        pass
    elif single_msg.type == "pitchwheel":
        # print(msg)
        msg_tokens.append(f"Pitchwheel_{single_msg.pitch}")
    else:
        raise Exception(f"Unknown message type: {single_msg.type}")
    return msg_tokens

def tokenize(midi):
    final = []
    final.append(f"TicksPerBeat{midi.ticks_per_beat}")
    for i, track in enumerate(midi.tracks):
        final.append(f"StartTrack_{i}")
        for msg in track:
            for token in msg_to_token(msg):
                final.append(token)
        final.append(f"EndTrack_{i}")
    return final