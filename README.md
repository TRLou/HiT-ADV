# Hide in Thicket: Generating Imperceptible and Rational Adversarial Perturbations on 3D Point Clouds  (CVPR 2024)

This work corresponds to the following paper: https://arxiv.org/abs/2403.05247.

This repo is based on [https://github.com/code-roamer/AOF.git] and [https://github.com/shikiw/SI-Adv.git]  to perform adversarial attack. 

![Framework](https://github.com/TRLou/HiT-ADV/assets/133848600/6e4f5b82-63fe-4084-a18c-a34e506d0e30)

```
@article{lou2024HiT-ADV,
  title={Hide in Thicket: Generating Imperceptible and Rational Adversarial Perturbations on 3D Point Clouds},
  author={Lou, Tianrui and Jia, Xiaojun and Gu, Jindong and Liu, Li and Liang, Siyuan and He, Bangyan and Cao, Xiaochun},
  journal={arXiv preprint arXiv:2403.05247},
  year={2024}
}
```
# Get Started
Step 1. Create a conda environment or use your existing one.
```
conda create --name hitadv python=3.8 -y
conda activate hitadv
pip install -r requirements.txt
```
Step 2. Prepare datasets and pretrained models.
Download from Baidu Yun：https://pan.baidu.com/s/1SL5-TuT9n74x5mADSM2E9g
(password:eaic)

Step 3. Evaluating：
```
python eval.py
```
Visualizing:
```
python visual.py
```


