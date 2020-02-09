import torch
from models.utils import smooth_predictions, concordance_cc2_np
import numpy as np


if __name__ == '__main__':
    x = torch.load('predictions_val.pt', map_location='cpu')
    gt_v, gt_a, pred_v, pred_a = x['valence_gt'], x['arousal_gt'], x['valence_pred'], x['arousal_pred']
    names = list(gt_v.keys())
    all_v_gt, all_a_gt, all_v_pred, all_a_pred = [], [], [], []
    for name in names:
        pv = smooth_predictions(pred_v[name])
        pa = smooth_predictions(pred_a[name])
        gv = gt_v[name].numpy()
        ga = gt_a[name].numpy()
        valid = (gv >= -1) & (ga >= -1)
        all_v_pred.append(pv[valid])
        all_a_pred.append(pa[valid])
        all_v_gt.append(gv[valid])
        all_a_gt.append(ga[valid])

    print (concordance_cc2_np(np.concatenate(all_v_pred), np.concatenate(all_v_gt)))
    print (concordance_cc2_np(np.concatenate(all_a_pred), np.concatenate(all_a_gt)))
