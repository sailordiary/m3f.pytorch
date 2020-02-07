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
            new_key = k.replace('v2p', 'shared')
            ckpt['state_dict'][new_key] = w
            print ('Created {}'.format(new_key))
        else:
            param_name = k.split('.')[-1]
            new_key_v = 'visual.v_private.{}.{}'.format(layer_id-12, param_name)
            new_key_a = 'visual.a_private.{}.{}'.format(layer_id-12, param_name)
            ckpt['state_dict'][new_key_v] = w
            ckpt['state_dict'][new_key_a] = w
            print ('Created {}, {}'.format(new_key_v, new_key_a))
    torch.save(ckpt, 'video_checkpoint.pt')
    print ('Successfully saved pre-trained video checkpoint!')
