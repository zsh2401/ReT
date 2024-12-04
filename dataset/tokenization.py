# 早期个人设计的MIDI Tokenizer....没什么用，放弃吧！
__time_to_tokens_cache = dict()
def time_to_tokens(x:int,shared_ttt_cache=None):
    
    global __time_to_tokens_cache
    key = f"{x}"

    cache = shared_ttt_cache if shared_ttt_cache is not None else __time_to_tokens_cache
    
    # if shared_ttt_cache is not None:
    #     print(cache)
        
    if key in cache:
        print("命中")
        return cache[key]
    
    coins = [1,5,10,20,50,100,200,500,1000]
    dp = [float('inf')] * (x + 1)
    dp[0] = 0
    # 记录路径
    path = [-1] * (x + 1)
    
    for j in range(1, x + 1):
        for c in coins:
            if j >= c and dp[j] > dp[j - c] + 1:
                dp[j] = dp[j - c] + 1
                path[j] = c  # 记录选择的硬币
    
    # 无法凑成金额 x
    if dp[x] == float('inf'):
        return -1, []
    
    # 回溯路径
    combination = []
    while x > 0:
        combination.append(path[x])
        x -= path[x]
    
    result_tokens = []
    for c in combination:
        result_tokens.append(f"Wait_{c}")
    __time_to_tokens_cache[key] = result_tokens
    return result_tokens

def msg_to_token(single_msg,shared_ttt_cache=None):
    msg_tokens = []
    # msg_tokens = []
    if single_msg.time > 0:
        if single_msg.time > 1000:
            print(single_msg)
        # print(single_msg.time )
        msg_tokens.extend(time_to_tokens(single_msg.time,shared_ttt_cache))
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

def tokenize(midi,shared_ttt_cache=None):
    final = []
    final.append(f"TicksPerBeat{midi.ticks_per_beat}")
    for i, track in enumerate(midi.tracks):
        final.append(f"StartTrack_{i}")
        for msg in track:
            for token in msg_to_token(msg,shared_ttt_cache):
                final.append(token)
        final.append(f"EndTrack_{i}")
    return final