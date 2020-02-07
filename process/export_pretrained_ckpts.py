import sys
from collections import OrderedDict

import torch


if __name__ == '__main__':
    # pretrained V2P (model.v2p)
    video_ckpt = torch.load(sys.argv[1])
    ckpt = OrderedDict()
    ckpt['state_dict'] = OrderedDict()
    for k, w in video_ckpt['state_dict'].items():
        if k.startswith('visual.fc'): continue
        layer_id = int(k.split('.')[2])
        if layer_id < 12: # up to conv3
            ckpt['state_dict'][k.replace('v2p', 'shared')] = w
        else:
            ckpt['state_dict']['visual.v_private.{}'.format(layer_id-12)] = w
            ckpt['state_dict']['visual.a_private.{}'.format(layer_id-12)] = w
    torch.save(ckpt, 'video_checkpoint.pt')
    print ('Successfully saved pre-trained video checkpoint!')
