observation space:
speaker: agent observation space Box(-inf, inf, (3,), float32)
listener: agent observation space Box(-inf, inf, (11,), float32)


observation is a dictionary key:speaker, value:array:
initial observation: {'speaker_0': array([0.15, 0.15, 0.65], dtype=float32), 'listener_0': array([ 0.        ,  0.        , -0.0208854 ,  0.56609595, -0.03218134,
        1.2176025 ,  0.47591156,  0.79811704,  0.        ,  0.        ,
        0.        ], dtype=float32)} debug info: {'speaker_0': {}, 'listener_0': {}}


clamp the action so that it is within the bound 0 and 1