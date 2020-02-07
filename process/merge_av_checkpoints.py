import sys
from collections import OrderedDict

import torch


if __name__ == '__main__':
    # pretrained GRU
    audio_ckpt = torch.load(sys.argv[1])
    # pretrained V2P (model.v2p)
    video_ckpt = torch.load(sys.argv[2])
    ckpt = OrderedDict()
    ckpt['state_dict'] = OrderedDict()
    for k, w in audio_ckpt['state_dict'].items():
        # drop fc layers
        if k.startswith('audio.fc'): continue
        else: ckpt['state_dict'][k] = w
    for k, w in video_ckpt['state_dict'].items():
        if k.startswith('visual.gru_a.fc'): continue
        elif k.startswith('visual.gru_v.fc'): continue
        else: ckpt['state_dict'][k] = w
    torch.save(ckpt, 'fused_av.pt')
    print ('Saved merged A-V checkpoint!')
