
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

## Dataset
We use the [Aff-Wild2 dataset](https://ibug.doc.ic.ac.uk/resources/aff-wild2/). The raw videos are decoded with ``ffmpeg", and passed to [RetinaFace-ResNet50](https://github.com/deepinsight/insightface/tree/master/RetinaFace) for face detection.

You can download our version of cropped-aligned face tracks here:

**OneDrive**: [[256x256 px]](https://mailsucaseducn-my.sharepoint.com/:f:/g/personal/zhangyuanhang15_mails_ucas_edu_cn/ErGo36iyXzFFtHcyXIQIuZABnaLsMiHE1CZ5EhsQ7HzhMw?e=9xBNXT)

(At this moment some files are still being uploaded, we apologize for the inconvenience.)

### Citation   
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
