
---   
<div align="center">    
 
# M³T: Multi-Modal Multi-Task Learning for Continuous Valence-Arousal Estimation

[![Paper](http://img.shields.io/badge/paper-arxiv.2002.02957-B31B1B.svg)](https://arxiv.org/abs/2002.02957)
[![Conference Workshop](http://img.shields.io/badge/FG-2020-4b44ce.svg)](https://ibug.doc.ic.ac.uk/resources/affect-recognition-wild-unimulti-modal-analysis-va/) 
[![Challenge](http://img.shields.io/badge/ABAW-2020-4b44ce.svg)](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/)   
</div>
 
## Description
Valence-arousal estimation models trained on Aff-Wild2.

## How to run   
First, install dependencies
```bash
# clone project   
git clone https://github.com/sailordiary/affwild2-va-models
python3 -m pip install -r requirements.txt --user
```

To evaluate on our pretrained models, first download the checkpoints from the release page, and run `eval.py` to generate val or test predictions:
```bash
# download the checkpoint
wget 
# to report CCC on the validation set
python3 eval.py --test_on_val --checkpoint m3t_mtl-vox2.pt
python3 get_smoothed_ccc predictions_val.pt
# to generate test predictions
python3 eval.py --checkpoint m3t_mtl-vox2.pt
```

## Dataset
We use the [Aff-Wild2 dataset](https://ibug.doc.ic.ac.uk/resources/aff-wild2/). The raw videos are decoded with `ffmpeg`, and passed to [RetinaFace-ResNet50](https://github.com/deepinsight/insightface/tree/master/RetinaFace) for face detection. To extract log-Mel spectrogram energies, extract 16kHz mono wave files from audio tracks, and refer to `process/extract_melspec.py`.

We provide the cropped-aligned face tracks (256x256) as well as pre-computed SENet-101 and TCAE features we use for our experiments here: [[OneDrive link]](https://mailsucaseducn-my.sharepoint.com/:f:/g/personal/zhangyuanhang15_mails_ucas_edu_cn/ErGo36iyXzFFtHcyXIQIuZABnaLsMiHE1CZ5EhsQ7HzhMw?e=sko5Uy)

Please note that in addition to the 256-dimensional encoder features, we also saved 12 AU activation scores predicted by TCAE, which together are concatenated into a 268-dimensional vector for each video frame. We only used the encoder features for our experiments, but feel free to experiment with this extra information.

## Citation   
```
@misc{zhang2020m3t,
    title={$M^3$T: Multi-Modal Continuous Valence-Arousal Estimation in the Wild},
    author={Yuan-Hang Zhang and Rulin Huang and Jiabei Zeng and Shiguang Shan and Xilin Chen},
    year={2020},
    eprint={2002.02957},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
